"""qlab — command-line interface.

Typical session:
    qlab init --capital 1000000
    qlab add SBER
    qlab add SU26238RMFS4          # ОФЗ 26238
    qlab candles SBER --from 2024-01-01
    qlab rates load key_rate.csv
    qlab buy SBER 100
    qlab loan take 300000 --spread 0.05
    qlab tick                       # advance to today, accrue everything
    qlab portfolio
    qlab ledger
"""
from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select, asc

from . import db
from .models import Account, Instrument, Candle, Order, Fill
from .core import ledger, execution, financing, accrual, portfolio, margin as margin_mod
from .moex.iss import ISSClient
from .moex import ingest

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
con = Console()

ACCOUNT = "main"


def _account(s) -> Account:
    acc = s.execute(select(Account).where(Account.name == ACCOUNT)).scalar_one_or_none()
    if acc is None:
        con.print("[red]Нет счёта. Сначала: qlab init[/red]")
        raise typer.Exit(1)
    return acc


def _instrument(s, secid: str) -> Instrument:
    ins = s.execute(select(Instrument).where(Instrument.secid == secid)).scalar_one_or_none()
    if ins is None:
        con.print(f"[red]{secid} не добавлен. Сначала: qlab add {secid}[/red]")
        raise typer.Exit(1)
    return ins


def _marks(s, *, refresh: bool) -> dict[int, Decimal]:
    """Prices for all instruments with open positions (and held cash refs)."""
    prices: dict[int, Decimal] = {}
    instruments = list(s.execute(select(Instrument)).scalars())
    if refresh:
        client = ISSClient()
        by_route: dict[tuple, list[Instrument]] = {}
        for ins in instruments:
            by_route.setdefault(ins.board, []).append(ins)
        for board, group in by_route.items():
            from .moex.iss import BOARD_ROUTE
            engine, market = BOARD_ROUTE[board]
            quotes = client.marketdata(engine, market, board, [i.secid for i in group])
            for ins in group:
                if ins.secid in quotes:
                    px = quotes[ins.secid]
                    if ins.asset_class in ("bond", "ofz") and ins.face_value:
                        px = px / 100 * ins.face_value     # % of par -> RUB
                    prices[ins.id] = px
    for ins in instruments:
        if ins.id not in prices:
            px = ingest.last_close(s, ins)
            if px is not None:
                if ins.asset_class in ("bond", "ofz") and ins.face_value:
                    px = px / 100 * ins.face_value
                prices[ins.id] = px
    return prices


@app.command()
def init(capital: int = typer.Option(1_000_000, help="Начальный капитал, RUB")):
    """Создать БД и счёт с фиксированным капиталом (без кнопки reset)."""
    db.init_db()
    with db.session() as s:
        if s.execute(select(Account).where(Account.name == ACCOUNT)).scalar_one_or_none():
            con.print("[yellow]Счёт уже существует. Reset не предусмотрен — это фича.[/yellow]")
            raise typer.Exit(1)
        now = datetime.now()
        acc = Account(name=ACCOUNT, sim_time=now, created_at=now)
        s.add(acc)
        s.flush()
        ledger.post_batch(s, acc.id, now, [
            ledger.Entry("cash", amount=Decimal(capital)),
            ledger.Entry("equity", amount=-Decimal(capital), memo="initial capital"),
        ], ref_type="init")
    con.print(f"[green]Счёт создан: {capital:,} ₽[/green]")


@app.command()
def add(secid: str, board: str = typer.Option(None)):
    """Добавить инструмент с MOEX (метаданные + купонное расписание для облигаций)."""
    with db.session() as s:
        ins = ingest.add_instrument(s, ISSClient(), secid, board)
        con.print(f"[green]{ins.secid}[/green] {ins.name} | {ins.board} | "
                  f"{ins.asset_class} | лот {ins.lot_size}")


@app.command()
def candles(secid: str,
            from_: str = typer.Option(None, "--from"),
            to: str = typer.Option(None, "--to"),
            tf: str = typer.Option("1d")):
    """Загрузить свечи с ISS."""
    with db.session() as s:
        ins = _instrument(s, secid)
        n = ingest.ingest_candles(
            s, ISSClient(), ins, tf,
            date.fromisoformat(from_) if from_ else None,
            date.fromisoformat(to) if to else None)
    con.print(f"загружено {n} свечей {tf}")


@app.command()
def buy(secid: str, qty: int, offline: bool = typer.Option(False)):
    _trade("buy", secid, qty, offline)


@app.command()
def sell(secid: str, qty: int, offline: bool = typer.Option(False)):
    _trade("sell", secid, qty, offline)


def _trade(side: str, secid: str, qty: int, offline: bool):
    with db.session() as s:
        acc = _account(s)
        ins = _instrument(s, secid)
        prices = _marks(s, refresh=not offline)
        if ins.id not in prices:
            con.print("[red]Нет цены — загрузите свечи или уберите --offline[/red]")
            raise typer.Exit(1)
        try:
            res = execution.execute_market_order(
                s, acc, ins, side, Decimal(qty), prices[ins.id], prices,
                ts=datetime.now() if acc.mode == "live_paper" else acc.sim_time)
        except execution.OrderRejected as e:
            con.print(f"[red]Отклонено: {e}[/red]")
            raise typer.Exit(1)
        f = res.fill
        con.print(f"[green]{side} {qty} {secid} @ {f.price:.4f}[/green] "
                  f"(ref {f.ref_price}, комиссия {f.commission:.2f}"
                  + (f", реализовано {res.realized_pnl:+.2f}" if res.realized_pnl else "") + ")")


loan_app = typer.Typer()
app.add_typer(loan_app, name="loan")


@loan_app.command("take")
def loan_take(amount: int,
              rate: float = typer.Option(None, help="фиксированная годовая, напр. 0.21"),
              spread: float = typer.Option(None, help="спред к ключевой, напр. 0.05")):
    if (rate is None) == (spread is None):
        con.print("[red]Укажите ровно одно: --rate или --spread[/red]")
        raise typer.Exit(1)
    model = ({"type": "fixed", "rate": str(rate)} if rate is not None
             else {"type": "key_rate", "spread": str(spread)})
    with db.session() as s:
        acc = _account(s)
        loan = financing.take_loan(s, acc, Decimal(amount), model)
    con.print(f"[green]Займ {amount:,} ₽, модель {model}[/green] id={loan.id[:8]}")


@loan_app.command("repay")
def loan_repay(loan_prefix: str, amount: int):
    with db.session() as s:
        acc = _account(s)
        from .models import Loan
        loan = next((l for l in s.execute(select(Loan).where(
            Loan.account_id == acc.id, Loan.status == "open")).scalars()
            if l.id.startswith(loan_prefix)), None)
        if loan is None:
            con.print("[red]Займ не найден[/red]"); raise typer.Exit(1)
        paid = financing.repay(s, acc, loan, Decimal(amount))
    con.print(f"[green]Погашено {paid:,} ₽[/green]")


rates_app = typer.Typer()
app.add_typer(rates_app, name="rates")


@rates_app.command("load")
def rates_load(path: Path):
    """CSV `date,percent` для ключевой ставки."""
    with db.session() as s:
        n = ingest.load_key_rate_csv(s, path.read_text())
    con.print(f"загружено {n} фиксингов key_rate")


@app.command()
def tick(to: str = typer.Option(None, help="YYYY-MM-DD; default: сегодня")):
    """Продвинуть часы счёта: проценты, купоны, погашения, маржа."""
    with db.session() as s:
        acc = _account(s)
        target = datetime.combine(date.fromisoformat(to), datetime.min.time()) \
            if to else datetime.now()
        accrual.tick_to(s, acc, target)
        prices = _marks(s, refresh=False)
        try:
            rep = margin_mod.update_account_state(s, acc, prices)
            state = rep.state
        except KeyError:
            state = "(нет цен для оценки маржи)"
    con.print(f"часы: {target:%Y-%m-%d}, маржинальный статус: {state}")


@app.command("portfolio")
def portfolio_cmd(offline: bool = typer.Option(False)):
    with db.session() as s:
        acc = _account(s)
        prices = _marks(s, refresh=not offline)
        view = portfolio.snapshot(s, acc, prices)

    t = Table(title=f"Портфель — {acc.sim_time:%Y-%m-%d}")
    for col in ("SECID", "qty", "avg", "last", "value", "uPnL", "НКД"):
        t.add_column(col, justify="right")
    for p in view.positions:
        t.add_row(p.secid, f"{p.qty:.0f}", f"{p.avg_price:.4f}", f"{p.last:.4f}",
                  f"{p.value:,.2f}", f"{p.upnl:+,.2f}", f"{p.aci:,.2f}")
    con.print(t)
    m = view.margin
    con.print(f"Кэш {view.cash:,.2f} | Долг {view.debt:,.2f} | NAV {view.nav:,.2f}")
    con.print(f"Стоимость портфеля {m.portfolio_value:,.2f} | НМ {m.initial_margin:,.2f} | "
              f"ММ {m.minimal_margin:,.2f} | статус [bold]{m.state}[/bold]")
    for l in view.loans:
        con.print(f"  займ {l.id[:8]}: {l.principal:,.2f} ₽, {l.rate_model}")


@app.command("ledger")
def ledger_cmd(n: int = 30):
    with db.session() as s:
        acc = _account(s)
        rows = ledger.ledger(s, acc.id, n)
    t = Table(title="Леджер (последние записи)")
    for col in ("ts", "leg", "instr", "qty", "amount", "memo"):
        t.add_column(col)
    with db.session() as s:
        for e in rows:
            ins = s.get(Instrument, e.instrument_id) if e.instrument_id else None
            t.add_row(f"{e.ts:%m-%d %H:%M}", e.leg, ins.secid if ins else "",
                      f"{e.qty:.0f}" if e.qty else "", f"{e.amount:+,.2f}",
                      e.memo or "")
    con.print(t)


def _load_closes(s, secid: str, tf: str = "1d") -> tuple[list[float], Instrument]:
    """Load candle close prices for a symbol from the local DB."""
    ins = _instrument(s, secid)
    rows = list(s.execute(
        select(Candle)
        .where(Candle.instrument_id == ins.id, Candle.tf == tf)
        .order_by(Candle.ts.asc())
    ).scalars())
    if not rows:
        con.print(f"[red]Нет свечей {tf} для {secid}. Сначала: qlab candles {secid}[/red]")
        raise typer.Exit(1)
    return [float(r.c) for r in rows], ins


def _log_returns(closes: list[float]) -> list[float]:
    import math
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


# ──────────────────────────────────────── lab subcommands ────────────────────

lab_app = typer.Typer(no_args_is_help=True)
app.add_typer(lab_app, name="lab", help="Аналитические лаборатории (стационарность, ARIMA, GARCH, коинтеграция)")


@lab_app.command("stat")
def lab_stat(
    secid: str,
    tf: str = typer.Option("1d", help="Таймфрейм свечей"),
    reg: str = typer.Option("c", help="Regression: c | ct | n"),
):
    """ADF / KPSS / Phillips-Perron тест стационарности."""
    from .labs import stationarity

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    result = stationarity.test(closes, regression=reg)

    # ── таблица тестов ──
    t = Table(title=f"Стационарность {ins.secid} ({len(closes)} баров {tf})",
              show_header=True)
    for col in ("Тест", "Стат.", "p-value", "Крит. 5%", "Результат"):
        t.add_column(col, justify="right" if col not in ("Тест", "Результат") else "left")

    def _row(name: str, r: stationarity.SingleTestResult) -> tuple:
        crit5 = r.critical.get("5%", float("nan"))
        verdict = ("[green]H₀ отвергнута[/green]" if r.reject_h0
                   else "[yellow]H₀ не отвергнута[/yellow]")
        return name, f"{r.stat:.4f}", f"{r.pvalue:.4f}", f"{crit5:.4f}", verdict

    t.add_row(*_row("ADF", result.adf))
    t.add_row(*_row("KPSS", result.kpss))
    t.add_row(*_row("PP", result.pp))
    con.print(t)

    # ── итог ──
    colour = {"stationary": "green", "non-stationary": "red", "ambiguous": "yellow"}[result.conclusion]
    con.print(f"\n[{colour}]Итог: {result.conclusion}[/{colour}]  "
              f"→  рекомендуется d=[bold]{result.d_suggested}[/bold]")

    # ── interpretation ──
    interp = result.interpretation
    con.print(f"\n[dim]ADF:[/dim]  {interp['adf']}")
    con.print(f"[dim]KPSS:[/dim] {interp['kpss']}")
    con.print(f"[dim]PP:[/dim]   {interp['pp']}")
    con.print(f"\n{interp['conclusion']}")
    for w in interp["caveats"]:
        con.print(f"  [dim]⚠  {w}[/dim]")


@lab_app.command("arima")
def lab_arima(
    secid: str,
    tf: str = typer.Option("1d"),
    p_max: int = typer.Option(5, help="Максимальный порядок p"),
    q_max: int = typer.Option(5, help="Максимальный порядок q"),
    d: str = typer.Option("auto", help="Порядок интегрирования: число или 'auto'"),
    criterion: str = typer.Option("aic", help="Критерий отбора: aic | bic"),
):
    """Grid search ARIMA(p,d,q) + walk-forward RMSE."""
    from .labs import arima as arima_lab, stationarity

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    if d == "auto":
        con.print("[dim]Определяю порядок дифференцирования...[/dim]")
        d_int = stationarity.test(closes).d_suggested
        con.print(f"[dim]d={d_int} (из теста стационарности)[/dim]")
    else:
        d_int = int(d)

    con.print(f"[dim]Grid search ARIMA(0–{p_max}, {d_int}, 0–{q_max}) по {criterion.upper()}...[/dim]")
    result = arima_lab.fit(closes, d=d_int, p_max=p_max, q_max=q_max, criterion=criterion)

    p, di, q = result.order
    con.print(f"\n[bold green]ARIMA({p},{di},{q})[/bold green]  "
              f"AIC={result.aic:.2f}  BIC={result.bic:.2f}  "
              f"LogL={result.log_likelihood:.2f}")
    con.print(f"In-sample: {result.n_insample} баров | Out-of-sample: {result.n_outsample} баров")

    # параметры
    pt = Table(title="Параметры модели")
    pt.add_column("Параметр"); pt.add_column("Значение", justify="right")
    for k, v in result.params.items():
        pt.add_row(k, f"{v:.6f}")
    con.print(pt)

    interp = result.interpretation
    lb_colour = "green" if result.ljung_box_p >= 0.05 else "red"
    arch_colour = "red" if (not __import__("math").isnan(result.arch_lm_p) and result.arch_lm_p < 0.05) else "green"
    con.print(f"\nLjung-Box p=[{lb_colour}]{result.ljung_box_p:.4f}[/{lb_colour}]  "
              f"ARCH-LM p=[{arch_colour}]{result.arch_lm_p:.4f}[/{arch_colour}]")
    if result.walk_forward_rmse is not None:
        con.print(f"Walk-forward RMSE: {result.walk_forward_rmse:.6f}")

    con.print(f"\n{interp['residuals']}")
    con.print(f"{interp['arch_effect']}")
    if "walk_forward" in interp:
        con.print(f"{interp['walk_forward']}")
    for w in interp["caveats"]:
        con.print(f"  [dim]⚠  {w}[/dim]")


@lab_app.command("garch")
def lab_garch(
    secid: str,
    tf: str = typer.Option("1d"),
    spec: str = typer.Option("garch", help="Спецификация: garch | egarch | gjr-garch"),
    compare: bool = typer.Option(False, help="Сравнить все три спецификации"),
):
    """GARCH(1,1) / EGARCH / GJR-GARCH на дневных лог-доходностях."""
    from .labs import garch as garch_lab

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    rets = _log_returns(closes)
    con.print(f"[dim]{len(rets)} лог-доходностей {ins.secid} ({tf})[/dim]")

    if compare:
        results = garch_lab.fit_all(rets)
        ct = Table(title="Сравнение спецификаций GARCH")
        for col in ("Spec", "AIC", "BIC", "σ₁ (%)", "σ₂₂ (%)", "σ_год (%)"):
            ct.add_column(col, justify="right" if col != "Spec" else "left")
        best_aic = min(r.aic for r in results.values())
        for sp, r in results.items():
            marker = " ★" if r.aic == best_aic else ""
            ct.add_row(sp + marker,
                       f"{r.aic:.2f}", f"{r.bic:.2f}",
                       f"{r.vol_forecast_1*100:.3f}",
                       f"{r.vol_forecast_22*100:.3f}",
                       f"{r.annualized_vol*100:.1f}")
        con.print(ct)
        spec = min(results, key=lambda s: results[s].aic)
        con.print(f"\n[dim]Детальный вывод для лучшей модели: {spec}[/dim]")
        result = results[spec]
    else:
        result = garch_lab.fit(rets, spec=spec)

    interp = result.interpretation
    con.print(f"\n[bold green]{result.spec.upper()}({result.p},{result.q})[/bold green]  "
              f"AIC={result.aic:.2f}  BIC={result.bic:.2f}")

    pt = Table(title="Параметры")
    pt.add_column("Параметр"); pt.add_column("Значение", justify="right")
    for k, v in result.params.items():
        pt.add_row(k, f"{v:.6f}")
    con.print(pt)

    con.print(f"\n{interp['volatility']}")
    if "persistence" in interp:
        con.print(interp["persistence"])
    if "leverage" in interp:
        lev_colour = "green" if result.has_leverage else "dim"
        con.print(f"[{lev_colour}]{interp['leverage']}[/{lev_colour}]")
    if "volatility_targeting" in interp:
        con.print(f"\n{interp['volatility_targeting']}")
    for w in interp["caveats"]:
        con.print(f"  [dim]⚠  {w}[/dim]")


@lab_app.command("coint")
def lab_coint(
    secid1: str,
    secid2: str,
    tf: str = typer.Option("1d"),
    method: str = typer.Option("eg", help="Метод: eg (Engle-Granger) | johansen"),
    kalman: bool = typer.Option(False, help="Дополнительно: Kalman hedge ratio"),
):
    """Коинтеграция пары инструментов."""
    from .labs import cointegration as coint_lab

    with db.session() as s:
        closes1, ins1 = _load_closes(s, secid1, tf)
        closes2, ins2 = _load_closes(s, secid2, tf)

    n = min(len(closes1), len(closes2))
    if n < 20:
        con.print("[red]Недостаточно данных (нужно ≥20 совместных баров)[/red]")
        raise typer.Exit(1)
    y, x = closes1[-n:], closes2[-n:]
    con.print(f"[dim]{n} совместных баров {tf}: {ins1.secid} (Y) ~ {ins2.secid} (X)[/dim]")

    if method == "johansen":
        result = coint_lab.johansen([y, x])
        interp = result.interpretation
        con.print(f"\n[bold]Johansen trace test[/bold]")
        jt = Table()
        jt.add_column("H₀: ≤r векторов"); jt.add_column("Trace стат.", justify="right")
        jt.add_column("Крит. 5%", justify="right"); jt.add_column("Отвергается?")
        for i, (ts, cv) in enumerate(zip(result.trace_stats, result.critical_5pct)):
            reject = ts > cv
            colour = "green" if reject else "dim"
            jt.add_row(f"r≤{i}", f"{ts:.3f}", f"{cv:.3f}",
                       f"[{colour}]{'да' if reject else 'нет'}[/{colour}]")
        con.print(jt)
        con.print(f"\n{interp['result']}")
        for w in interp["caveats"]:
            con.print(f"  [dim]⚠  {w}[/dim]")
    else:
        result_eg = coint_lab.engle_granger(y, x)
        interp = result_eg.interpretation
        eg_colour = "green" if result_eg.cointegrated else "red"
        con.print(f"\n[bold]Engle-Granger[/bold] stat={result_eg.stat:.4f}  "
                  f"p=[{eg_colour}]{result_eg.pvalue:.4f}[/{eg_colour}]")
        con.print(f"OLS hedge ratio β={result_eg.hedge_ratio_ols:.4f}")
        con.print(f"\n{interp['coint_test']}")
        con.print(interp["hedge_ratio"])
        con.print(interp["mean_reversion"])
        con.print(f"\n[dim]{interp['trading_signal']}[/dim]")
        for w in interp["caveats"]:
            con.print(f"  [dim]⚠  {w}[/dim]")

    if kalman:
        con.print("\n[bold]Kalman hedge ratio[/bold]")
        kr = coint_lab.kalman_hedge(y, x)
        interp_k = kr.interpretation
        beta_arr = kr.hedge_ratios
        con.print(f"β: min={min(beta_arr):.4f}  max={max(beta_arr):.4f}  "
                  f"mean={sum(beta_arr)/len(beta_arr):.4f}")
        con.print(interp_k["hedge_ratio"])
        con.print(interp_k["spread"])
        con.print(f"\n[dim]{interp_k['trading_signal']}[/dim]")
        for w in interp_k["caveats"]:
            con.print(f"  [dim]⚠  {w}[/dim]")


# ─────────────────────────────────────── general info commands ───────────────

@app.command("instruments")
def instruments_cmd():
    """Список всех инструментов в базе."""
    with db.session() as s:
        rows = list(s.execute(select(Instrument).order_by(Instrument.secid)).scalars())
    if not rows:
        con.print("[yellow]Нет инструментов. Добавьте: qlab add SBER[/yellow]")
        return
    t = Table(title="Инструменты")
    for col in ("SECID", "Название", "Борд", "Класс", "Лот", "Спред bps", "Комис. bps"):
        t.add_column(col)
    for ins in rows:
        t.add_row(ins.secid, (ins.name or "")[:40], ins.board, ins.asset_class,
                  str(ins.lot_size), str(ins.spread_bps), str(ins.commission_bps))
    con.print(t)


@app.command("orders")
def orders_cmd(
    n: int = typer.Option(30, help="Количество записей"),
    status: str = typer.Option("all", help="Фильтр: all | filled | new | cancelled"),
):
    """История ордеров счёта."""
    with db.session() as s:
        acc = _account(s)
        q = select(Order).where(Order.account_id == acc.id).order_by(Order.created_ts.desc())
        if status != "all":
            q = q.where(Order.status == status)
        rows = list(s.execute(q.limit(n)).scalars())
        if not rows:
            con.print("[yellow]Нет ордеров.[/yellow]")
            return
        t = Table(title=f"Ордера ({status})")
        for col in ("Дата", "SECID", "Сторона", "Тип", "Кол-во", "Лимит", "Статус"):
            t.add_column(col)
        for o in rows:
            ins = s.get(Instrument, o.instrument_id)
            t.add_row(
                f"{o.created_ts:%Y-%m-%d %H:%M}",
                ins.secid if ins else str(o.instrument_id),
                o.side, o.order_type,
                f"{o.qty:.0f}",
                f"{o.limit_price:.4f}" if o.limit_price else "—",
                o.status)
        con.print(t)


@app.command("pnl")
def pnl_cmd():
    """Реализованный P&L по инструментам (из леджера)."""
    with db.session() as s:
        acc = _account(s)
        from .models import LedgerEntry
        rows = list(s.execute(
            select(LedgerEntry)
            .where(LedgerEntry.account_id == acc.id,
                   LedgerEntry.leg == "realized_pnl")
            .order_by(LedgerEntry.instrument_id)
        ).scalars())

    if not rows:
        con.print("[yellow]Нет реализованного P&L. Закройте хотя бы одну позицию.[/yellow]")
        return

    by_instr: dict[int | None, Decimal] = {}
    for e in rows:
        by_instr[e.instrument_id] = by_instr.get(e.instrument_id, Decimal(0)) - e.amount

    t = Table(title="Реализованный P&L")
    t.add_column("Инструмент"); t.add_column("P&L, ₽", justify="right")
    total = Decimal(0)
    with db.session() as s:
        for iid, pnl in sorted(by_instr.items(), key=lambda x: -abs(x[1])):
            ins = s.get(Instrument, iid) if iid else None
            name = ins.secid if ins else "—"
            colour = "green" if pnl >= 0 else "red"
            t.add_row(name, f"[{colour}]{pnl:+,.2f}[/{colour}]")
            total += pnl
    colour = "green" if total >= 0 else "red"
    t.add_section()
    t.add_row("[bold]ИТОГО[/bold]", f"[bold {colour}]{total:+,.2f}[/bold {colour}]")
    con.print(t)


# ─────────────────────────────────── extra lab commands ──────────────────────

@lab_app.command("ou")
def lab_ou(
    secid: str,
    tf: str = typer.Option("1d"),
    log_prices: bool = typer.Option(True, help="Фитировать на лог-ценах (False = raw)"),
):
    """Ornstein-Uhlenbeck процесс: θ, μ, σ, half-life."""
    from .labs.ou import estimate

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    import math
    x = [math.log(p) for p in closes] if log_prices else closes
    fit = estimate(x)

    con.print(f"\n[bold]OU-процесс: {ins.secid} ({len(x)} баров {tf}{'  лог-цены' if log_prices else ''})[/bold]")
    t = Table()
    t.add_column("Параметр"); t.add_column("Значение", justify="right"); t.add_column("Смысл")
    t.add_row("θ (speed)", f"{fit.theta:.6f}", "скорость возврата к μ")
    t.add_row("μ (mean)", f"{fit.mu:.6f}", "долгосрочный уровень")
    t.add_row("σ (vol)", f"{fit.sigma:.6f}", "мгновенная волатильность")
    t.add_row("half-life", f"{fit.half_life:.1f} шагов", "за сколько шагов отклонение ↓ вдвое")
    t.add_row("φ (AR1)", f"{fit.phi:.6f}", "AR(1)-коэффициент")
    t.add_row("n", str(fit.n), "число наблюдений")
    con.print(t)
    interp = fit.interpretation
    con.print(f"\n{interp['half_life']}")
    for w in interp.get("caveats", []):
        con.print(f"  [dim]⚠  {w}[/dim]")


@lab_app.command("returns")
def lab_returns(
    secid: str,
    tf: str = typer.Option("1d"),
    periods_per_year: int = typer.Option(252, help="Периодов в году (252=дневной)"),
):
    """Статистика доходностей: Sharpe, Sortino, drawdown, моменты распределения."""
    from .labs.stats import returns_summary

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    r = returns_summary(closes, periods_per_year=periods_per_year)

    con.print(f"\n[bold]Доходности: {ins.secid} ({r.n} баров {tf})[/bold]")
    t = Table()
    t.add_column("Метрика"); t.add_column("Значение", justify="right")

    def pct(v: float) -> str:
        return f"{v*100:+.2f}%"

    t.add_row("Суммарная доходность", pct(r.total_return))
    t.add_row("Годовая доходность (лог)", pct(r.annual_return))
    t.add_row("Дневная волатильность", pct(r.daily_vol))
    t.add_row("Годовая волатильность", pct(r.annual_vol))
    t.add_row("Sharpe (rf=0)", f"{r.sharpe:.3f}")
    t.add_row("Sortino (rf=0)", f"{r.sortino:.3f}")
    t.add_row("Макс. просадка", f"-{r.max_drawdown*100:.2f}%")
    t.add_row("Асимметрия (skew)", f"{r.skewness:.3f}")
    t.add_row("Эксцесс (excess kurt)", f"{r.kurtosis:.3f}")
    t.add_row("Лучший день", pct(r.best_day))
    t.add_row("Худший день", pct(r.worst_day))
    t.add_row("Дней в плюс", f"{r.positive_days_pct:.1f}%")
    con.print(t)

    interp = r.interpretation
    con.print(f"\n{interp['return_vol']}")
    con.print(f"{interp['drawdown']}")
    con.print(f"{interp['distribution']}")
    for w in interp["caveats"]:
        con.print(f"  [dim]⚠  {w}[/dim]")


@lab_app.command("acf")
def lab_acf(
    secid: str,
    tf: str = typer.Option("1d"),
    lags: int = typer.Option(20, help="Число лагов"),
    returns: bool = typer.Option(True, help="True=на лог-доходностях, False=на ценах"),
):
    """ACF и PACF с текстовым bar-чартом."""
    from .labs.stats import acf_pacf

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    if returns:
        import math
        series = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        series_label = "лог-доходности"
    else:
        series = closes
        series_label = "цены"

    result = acf_pacf(series, lags=lags)
    ci = result.ci_95

    con.print(f"\n[bold]ACF / PACF: {ins.secid} ({result.n} наблюдений, {series_label})[/bold]")
    con.print(f"95% CI: ±{ci:.4f}\n")

    bar_width = 30

    def _bar(v: float) -> str:
        filled = int(abs(v) / 1.0 * bar_width)  # scale: 1.0 = full bar
        filled = min(filled, bar_width)
        ci_filled = int(ci / 1.0 * bar_width)
        bar = "▓" * filled + "░" * (bar_width - filled)
        sign = "+" if v >= 0 else "-"
        sig = "[green]*[/green]" if abs(v) > ci else " "
        return f"{sign}{bar}  {v:+.4f} {sig}"

    t = Table(title="")
    t.add_column("Лаг", justify="right")
    t.add_column("ACF")
    t.add_column("PACF")

    for lag, av, pv in zip(result.lags, result.acf_values, result.pacf_values):
        t.add_row(str(lag), _bar(av), _bar(pv))
    con.print(t)

    interp = result.interpretation
    con.print(f"\n{interp['summary']}")
    con.print(f"{interp['pattern']}")
    for w in interp["caveats"]:
        con.print(f"  [dim]⚠  {w}[/dim]")


@lab_app.command("vol")
def lab_vol(
    secid: str,
    tf: str = typer.Option("1d"),
    window: int = typer.Option(21, help="Окно rolling vol (торговых периодов)"),
    periods_per_year: int = typer.Option(252),
):
    """Rolling и общая историческая волатильность."""
    import math

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(rets)
    if n < window + 1:
        con.print(f"[red]Нужно минимум {window + 1} доходностей, есть {n}[/red]")
        raise typer.Exit(1)

    import numpy as np
    overall_vol = float(np.std(rets, ddof=1)) * math.sqrt(periods_per_year)

    # rolling vol
    rolling_vols = []
    for i in range(window - 1, n):
        w_rets = rets[i - window + 1: i + 1]
        rolling_vols.append(float(np.std(w_rets, ddof=1)) * math.sqrt(periods_per_year))

    current_vol = rolling_vols[-1]
    min_vol = min(rolling_vols)
    max_vol = max(rolling_vols)
    pct_rank = sum(1 for v in rolling_vols if v < current_vol) / len(rolling_vols) * 100

    con.print(f"\n[bold]Волатильность: {ins.secid} ({n+1} баров {tf}, окно {window}д)[/bold]")
    t = Table()
    t.add_column("Метрика"); t.add_column("Значение", justify="right")
    t.add_row("Историческая vol (весь ряд)", f"{overall_vol*100:.2f}%")
    t.add_row(f"Rolling {window}д vol (текущая)", f"{current_vol*100:.2f}%")
    t.add_row(f"Rolling {window}д vol (минимум)", f"{min_vol*100:.2f}%")
    t.add_row(f"Rolling {window}д vol (максимум)", f"{max_vol*100:.2f}%")
    t.add_row("Текущий перцентиль", f"{pct_rank:.0f}%")
    con.print(t)

    # sparkline (20 последних точек)
    tail = rolling_vols[-20:]
    max_tail = max(tail) or 1
    spark_height = 5
    cols = []
    for v in tail:
        bars = round(v / max_tail * spark_height)
        col = ["  "] * spark_height
        for b in range(bars):
            col[spark_height - 1 - b] = "██"
        cols.append(col)

    con.print(f"\n[dim]Rolling {window}д vol (последние 20 периодов):[/dim]")
    for row in range(spark_height):
        line = "  ".join(c[row] for c in cols)
        con.print(f"  {line}")
    con.print(f"  {''.join(['▔▔  '] * len(tail))}")
    pct_str = f"{pct_rank:.0f}%"
    v_colour = "red" if pct_rank > 75 else "green" if pct_rank < 25 else "yellow"
    con.print(f"\n[{v_colour}]Текущая vol в {pct_rank:.0f}-м перцентиле истории[/{v_colour}]")


@lab_app.command("regime")
def lab_regime(
    secid: str,
    tf: str = typer.Option("1d"),
    n: int = typer.Option(2, help="Число режимов (2 или 3)"),
):
    """Markov regime-switching: bull/bear + вероятности режимов."""
    from .labs.regime import markov_switching

    with db.session() as s:
        closes, ins = _load_closes(s, secid, tf)

    rets = _log_returns(closes)
    con.print(f"[dim]Фитирую Markov({n}-режима) на {len(rets)} доходностях {ins.secid}...[/dim]")

    result = markov_switching(rets, n_regimes=n)

    con.print(f"\n[bold]Markov regime-switching: {ins.secid} ({n} режима)[/bold]  "
              f"AIC={result.aic:.1f}  BIC={result.bic:.1f}")

    t = Table()
    t.add_column("Режим")
    t.add_column("μ (день)", justify="right")
    t.add_column("σ (день)", justify="right")
    t.add_column("Длит. (дней)", justify="right")
    t.add_column("Тек. вероятность", justify="right")

    labels = ["медвежий", "нейтр.", "бычий"]
    for i in range(n):
        label = labels[i] if i < len(labels) else f"режим {i}"
        prob = result.current_probs[i]
        prob_colour = "green" if prob > 0.6 else "yellow" if prob > 0.3 else "dim"
        t.add_row(
            f"{i}: {label}",
            f"{result.regime_means[i]*100:+.3f}%",
            f"{result.regime_vols[i]*100:.3f}%",
            f"{result.expected_durations[i]:.0f}",
            f"[{prob_colour}]{prob*100:.1f}%[/{prob_colour}]")
    con.print(t)

    interp = result.interpretation
    con.print(f"\n{interp['current']}")
    con.print(f"\n[dim]{interp['trading_use']}[/dim]")
    for w in interp["caveats"]:
        con.print(f"  [dim]⚠  {w}[/dim]")


@lab_app.command("corr")
def lab_corr(
    secid1: str,
    secid2: str,
    tf: str = typer.Option("1d"),
    window: int = typer.Option(60, help="Окно rolling корреляции"),
):
    """Статическая и rolling корреляция двух инструментов."""
    import math
    import numpy as np

    with db.session() as s:
        closes1, ins1 = _load_closes(s, secid1, tf)
        closes2, ins2 = _load_closes(s, secid2, tf)

    nmin = min(len(closes1), len(closes2))
    rets1 = [math.log(closes1[-nmin:][i] / closes1[-nmin:][i - 1])
              for i in range(1, nmin)]
    rets2 = [math.log(closes2[-nmin:][i] / closes2[-nmin:][i - 1])
              for i in range(1, nmin)]

    r1 = np.array(rets1); r2 = np.array(rets2)
    n = len(r1)

    static_corr = float(np.corrcoef(r1, r2)[0, 1])

    if n < window + 1:
        con.print(f"[yellow]Недостаточно данных для rolling {window}д. Показываю только статику.[/yellow]")
        rolling_corrs: list[float] = []
    else:
        rolling_corrs = []
        for i in range(window - 1, n):
            w1 = r1[i - window + 1: i + 1]
            w2 = r2[i - window + 1: i + 1]
            rolling_corrs.append(float(np.corrcoef(w1, w2)[0, 1]))

    con.print(f"\n[bold]Корреляция: {ins1.secid} / {ins2.secid} ({tf}, {n} доходностей)[/bold]")

    t = Table()
    t.add_column("Метрика"); t.add_column("Значение", justify="right")
    corr_colour = ("green" if abs(static_corr) > 0.7 else
                   "yellow" if abs(static_corr) > 0.3 else "dim")
    t.add_row("Статическая корреляция", f"[{corr_colour}]{static_corr:+.4f}[/{corr_colour}]")
    if rolling_corrs:
        t.add_row(f"Rolling {window}д (текущая)", f"{rolling_corrs[-1]:+.4f}")
        t.add_row(f"Rolling {window}д (мин)", f"{min(rolling_corrs):+.4f}")
        t.add_row(f"Rolling {window}д (макс)", f"{max(rolling_corrs):+.4f}")
        t.add_row(f"Rolling {window}д (среднее)", f"{float(np.mean(rolling_corrs)):+.4f}")
    con.print(t)

    # interpretation
    if abs(static_corr) > 0.8:
        desc = "очень высокая — бумаги движутся практически синхронно"
    elif abs(static_corr) > 0.5:
        desc = "высокая — существенная совместная динамика"
    elif abs(static_corr) > 0.2:
        desc = "умеренная"
    else:
        desc = "низкая — бумаги движутся независимо"
    sign_desc = "положительная (одно направление)" if static_corr > 0 else "отрицательная (противоположные)"
    con.print(f"\nКорреляция {sign_desc}, {desc}.")

    if rolling_corrs:
        rolling_range = max(rolling_corrs) - min(rolling_corrs)
        if rolling_range > 0.4:
            con.print("[yellow]Корреляция нестабильна (диапазон >{:.2f}) — "
                      "статический hedge ratio ненадёжен.[/yellow]".format(rolling_range))
        else:
            con.print("[dim]Корреляция относительно стабильна.[/dim]")

    con.print(f"\n[dim]⚠  Корреляция ≠ коинтеграция: qlab lab coint {secid1} {secid2}[/dim]")


if __name__ == "__main__":
    app()
