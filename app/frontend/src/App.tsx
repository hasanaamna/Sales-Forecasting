import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  CircleAlert,
  Clock3,
  Download,
  Info,
  Layers3,
  LoaderCircle,
  Menu,
  Play,
  ShoppingCart,
  Sparkles,
  TrendingUp,
  Trophy,
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8000";

type HistoryPoint = {
  date: string;
  sales: number;
};

type ForecastPoint = {
  date: string;
  predicted_sales: number;
};

type Metric = {
  model: string;
  MAE?: number;
  RMSE?: number;
  WAPE?: number;
  wape?: number;
  WAPE_percent?: number;
  [key: string]: unknown;
};

const money = (value: number) =>
  new Intl.NumberFormat("en-US", {
    notation: value >= 1_000_000 ? "compact" : "standard",
    maximumFractionDigits: 2,
  }).format(value);

const fullNumber = (value: number) =>
  new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);

const shortDate = (value: string) =>
  new Date(`${value.slice(0, 10)}T00:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

function readArray<T>(payload: unknown, keys: string[]): T[] {
  if (Array.isArray(payload)) return payload as T[];

  if (payload && typeof payload === "object") {
    const data = payload as Record<string, unknown>;

    for (const key of keys) {
      if (Array.isArray(data[key])) return data[key] as T[];
    }
  }

  return [];
}

function App() {
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [horizon, setHorizon] = useState(7);
  const [period, setPeriod] = useState("All");
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const [activeTab, setActiveTab] = useState("Overview");
  const [mobileMenu, setMobileMenu] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setInitialLoading(true);

      try {
        const [historyResponse, metricsResponse] = await Promise.all([
          fetch(`${API}/history?days=0`),
          fetch(`${API}/model-metrics`),
        ]);

        if (!historyResponse.ok) {
          throw new Error(
            "Historical sales endpoint is not available yet. Complete Step 4 below.",
          );
        }

        const historyPayload = await historyResponse.json();
        const historyRows = readArray<Record<string, unknown>>(
          historyPayload,
          ["history", "data", "results"],
        );

        const parsedHistory = historyRows
          .map((row) => ({
            date: String(row.date ?? ""),
            sales: Number(row.sales ?? row.total_sales ?? 0),
          }))
          .filter((row) => row.date && Number.isFinite(row.sales))
          .sort((a, b) => a.date.localeCompare(b.date));

        let parsedMetrics: Metric[] = [];

        if (metricsResponse.ok) {
          const metricsPayload = await metricsResponse.json();
          parsedMetrics = readArray<Metric>(metricsPayload, [
            "metrics",
            "results",
            "data",
          ]);
        }

        if (!cancelled) {
          setHistory(parsedHistory);
          setMetrics(parsedMetrics);
          setApiOnline(true);
          setError("");
        }
      } catch (err) {
        if (!cancelled) {
          setApiOnline(false);
          setError(
            err instanceof Error
              ? err.message
              : "Could not connect to the forecasting API.",
          );
        }
      } finally {
        if (!cancelled) setInitialLoading(false);
      }
    }

    void loadDashboard();

    return () => {
      cancelled = true;
    };
  }, []);

  const visibleHistory = useMemo(() => {
    if (period === "All") return history;

    const days = period === "3M" ? 90 : period === "6M" ? 180 : 365;
    return history.slice(-days);
  }, [history, period]);

  const totalSales = useMemo(
    () => history.reduce((sum, row) => sum + row.sales, 0),
    [history],
  );

  const averageSales = history.length ? totalSales / history.length : 0;

  const highestDay = history.reduce<HistoryPoint | null>(
    (best, row) => (!best || row.sales > best.sales ? row : best),
    null,
  );

  const lowestDay = history.reduce<HistoryPoint | null>(
    (best, row) => (!best || row.sales < best.sales ? row : best),
    null,
  );

  const weekdayData = useMemo(() => {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const totals = days.map(() => ({ total: 0, count: 0 }));

    history.forEach((row) => {
      const day = new Date(`${row.date.slice(0, 10)}T00:00:00`).getDay();
      totals[day].total += row.sales;
      totals[day].count += 1;
    });

    return days.map((day, index) => ({
      day,
      average: totals[index].count
        ? totals[index].total / totals[index].count
        : 0,
    }));
  }, [history]);

  const monthlyData = useMemo(() => {
    const months = new Map<string, { total: number; count: number }>();

    history.forEach((row) => {
      const key = row.date.slice(0, 7);
      const current = months.get(key) ?? { total: 0, count: 0 };
      current.total += row.sales;
      current.count += 1;
      months.set(key, current);
    });

    return Array.from(months.entries())
      .slice(-12)
      .map(([month, value]) => ({
        month: new Date(`${month}-01T00:00:00`).toLocaleDateString("en-US", {
          month: "short",
        }),
        average: value.total / value.count,
      }));
  }, [history]);

  const bestWape = (metric: Metric) =>
    Number(metric.WAPE ?? metric.wape ?? metric.WAPE_percent ?? 0);

  async function generateForecast() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ horizon }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Forecast request failed.");
      }

      const payload = await response.json();
      const rows = readArray<Record<string, unknown>>(payload, [
        "forecast",
        "forecasts",
        "predictions",
        "data",
      ]);

      const parsed = rows
        .map((row) => ({
          date: String(row.date ?? row.ds ?? ""),
          predicted_sales: Number(
            row.predicted_sales ?? row.prediction ?? row.sales ?? row.yhat ?? 0,
          ),
        }))
        .filter(
          (row) =>
            row.date &&
            Number.isFinite(row.predicted_sales),
        );

      if (!parsed.length) {
        throw new Error(
          "The API responded, but the forecast format was not recognised. Check the JSON returned by /forecast.",
        );
      }

      setForecast(parsed);
      setApiOnline(true);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Forecast generation failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  function downloadForecast() {
    if (!forecast.length) return;

    const csv = [
      "date,predicted_sales",
      ...forecast.map(
        (row) => `${row.date},${row.predicted_sales.toFixed(2)}`,
      ),
    ].join("\n");

    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8;" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "sales_forecast.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  const forecastTotal = forecast.reduce(
    (sum, row) => sum + row.predicted_sales,
    0,
  );

  const forecastAverage = forecast.length
    ? forecastTotal / forecast.length
    : 0;

  const navItems = [
    { label: "Overview", icon: Layers3 },
    { label: "Forecast", icon: TrendingUp },
    { label: "Insights", icon: BarChart3 },
    { label: "Model Performance", icon: Activity },
  ];

  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          className="mobile-menu"
          aria-label="Toggle navigation"
          onClick={() => setMobileMenu(!mobileMenu)}
        >
          <Menu size={21} />
        </button>

        <a className="brand" href="#overview" aria-label="Nexus home">
          NEXUS<span className="brand-dot">.</span>
          <span className="brand-subtitle">Retail Analytics</span>
        </a>

        <nav className={`top-nav ${mobileMenu ? "nav-open" : ""}`}>
          {navItems.map(({ label, icon: Icon }) => (
            <a
              key={label}
              href={
                label === "Overview"
                  ? "#overview"
                  : label === "Forecast"
                    ? "#forecast"
                    : label === "Insights"
                      ? "#insights"
                      : "#performance"
              }
              className={`nav-link ${activeTab === label ? "active" : ""}`}
              onClick={() => {
                setActiveTab(label);
                setMobileMenu(false);
              }}
            >
              <Icon className="nav-mobile-icon" size={16} />
              {label}
            </a>
          ))}
        </nav>

        <div className="topbar-meta">
          <div className="updated">
            <CalendarDays size={18} />
            <span>
              <small>Dataset through</small>
              <strong>{history.length ? history[history.length - 1].date : "—"}</strong>
            </span>
          </div>
          <div className={`api-pill ${apiOnline ? "online" : "offline"}`}>
            <span className="status-dot" />
            {apiOnline ? "API Online" : "API Offline"}
          </div>
        </div>
      </header>

      <main id="overview" className="dashboard">
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              SALES FORECASTING
            </div>

            <h1>
              Turn historical
              <br />
              sales into tomorrow’s
              <br />
              <em>opportunities.</em>
            </h1>

            <div className="hero-description">
              <span className="orange-rule" />
              <p>
                Explore retail demand, uncover sales patterns, and forecast
                upcoming daily sales with your machine learning model.
              </p>
            </div>

            <div className="hero-footnote">
              <Sparkles size={15} />
              Powered by your trained Random Forest model
            </div>
          </div>

          <div className="hero-art" aria-label="Retail analytics illustration">
            <div className="art-circle" />
            <div className="art-label">
              <span>DATA INSIGHT</span>
              <strong>Better decisions.</strong>
              <i />
            </div>
            <div className="art-stat">
              <span>HISTORICAL DATA</span>
              <strong>2013 — 2017</strong>
            </div>
            <div className="art-bars">
              {[25, 38, 30, 48, 43, 59, 51, 73, 66, 88, 75, 100].map(
                (height, index) => (
                  <span key={index} style={{ height: `${height}%` }} />
                ),
              )}
            </div>
          </div>
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <CircleAlert size={19} />
            <span>{error}</span>
          </div>
        )}

        <section className="kpi-grid" aria-label="Historical sales indicators">
          <article className="kpi-card">
            <div className="kpi-icon mint">
              <ShoppingCart size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Total Sales</span>
              <span className="kpi-caption">Historical dataset</span>
              <strong>{initialLoading ? "…" : money(totalSales)}</strong>
              <small className="kpi-note">Across all available dates</small>
            </div>
            <div className="mini-bars green-bars">
              {[25, 34, 30, 42, 38, 55, 51, 67, 60, 82, 74, 100].map(
                (height, i) => (
                  <i key={i} style={{ height: `${height}%` }} />
                ),
              )}
            </div>
          </article>

          <article className="kpi-card">
            <div className="kpi-icon peach">
              <BarChart3 size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Average Daily Sales</span>
              <span className="kpi-caption">Daily average</span>
              <strong>{initialLoading ? "…" : fullNumber(averageSales)}</strong>
              <small className="kpi-note">Average across recorded days</small>
            </div>
            <div className="mini-bars orange-bars">
              {[22, 28, 40, 34, 50, 45, 65, 58, 75, 70, 84, 100].map(
                (height, i) => (
                  <i key={i} style={{ height: `${height}%` }} />
                ),
              )}
            </div>
          </article>

          <article className="kpi-card">
            <div className="kpi-icon lavender">
              <TrendingUp size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Highest Daily Sales</span>
              <span className="kpi-caption">Peak recorded day</span>
              <strong>{highestDay ? fullNumber(highestDay.sales) : "—"}</strong>
              <small className="kpi-note">
                {highestDay ? `On ${highestDay.date}` : "Waiting for data"}
              </small>
            </div>
            <ArrowUpRight className="kpi-corner-icon" size={21} />
          </article>

          <article className="kpi-card">
            <div className="kpi-icon sky">
              <ArrowDownRight size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Lowest Daily Sales</span>
              <span className="kpi-caption">Lowest recorded day</span>
              <strong>{lowestDay ? fullNumber(lowestDay.sales) : "—"}</strong>
              <small className="kpi-note">
                {lowestDay ? `On ${lowestDay.date}` : "Waiting for data"}
              </small>
            </div>
            <ArrowDownRight className="kpi-corner-icon blue-icon" size={21} />
          </article>
        </section>

        <section className="content-grid">
          <div className="main-column">
            <article className="panel history-panel">
              <div className="panel-heading">
                <div className="heading-icon">
                  <BarChart3 size={22} />
                </div>
                <div>
                  <h2>Historical Sales Trend</h2>
                  <p>Daily total sales across the available dataset</p>
                </div>
                <div className="period-selector">
                  {["3M", "6M", "1Y", "All"].map((value) => (
                    <button
                      key={value}
                      className={period === value ? "selected" : ""}
                      onClick={() => setPeriod(value)}
                    >
                      {value}
                    </button>
                  ))}
                </div>
              </div>

              <div className="chart-area">
                {initialLoading ? (
                  <div className="empty-state">
                    <LoaderCircle className="spin" size={24} />
                    Loading historical sales…
                  </div>
                ) : visibleHistory.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={visibleHistory}
                      margin={{ top: 12, right: 10, left: 0, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient
                          id="salesGradient"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop offset="0%" stopColor="#205b43" stopOpacity={0.22} />
                          <stop offset="95%" stopColor="#205b43" stopOpacity={0.01} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        stroke="#e9e7e0"
                        vertical
                        strokeDasharray="2 4"
                      />
                      <XAxis
                        dataKey="date"
                        tickFormatter={(value) => shortDate(String(value))}
                        tick={{ fill: "#7c817b", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        minTickGap={38}
                      />
                      <YAxis
                        tickFormatter={(value) => money(Number(value))}
                        tick={{ fill: "#7c817b", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        width={58}
                      />
                      <Tooltip
                        labelFormatter={(value) => String(value)}
                        formatter={(value) => [
                          fullNumber(Number(value)),
                          "Daily sales",
                        ]}
                        contentStyle={{
                          border: "1px solid #e6e1d7",
                          borderRadius: 12,
                          background: "#fffdf8",
                          fontSize: 12,
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="sales"
                        stroke="#205b43"
                        strokeWidth={1.8}
                        fill="url(#salesGradient)"
                        dot={false}
                        activeDot={{ r: 4, fill: "#e96b24", stroke: "#fff" }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state">
                    <BarChart3 size={25} />
                    Historical chart will appear when the API is connected.
                  </div>
                )}
              </div>
              <div className="chart-footer">
                <span>
                  <i className="legend-dot" /> Actual historical sales
                </span>
                <span>
                  {visibleHistory.length.toLocaleString()} daily observations
                </span>
              </div>
            </article>

            <section id="insights" className="insight-grid">
              <article className="panel small-chart-panel">
                <div className="small-panel-heading">
                  <div className="small-heading-icon orange-text">
                    <CalendarDays size={18} />
                  </div>
                  <div>
                    <h3>Sales by Day of Week</h3>
                    <p>Average daily sales</p>
                  </div>
                </div>
                <div className="small-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={weekdayData} margin={{ top: 12, right: 2, left: -20, bottom: 0 }}>
                      <CartesianGrid vertical={false} stroke="#ece8df" />
                      <XAxis dataKey="day" tick={{ fontSize: 10, fill: "#777e78" }} axisLine={false} tickLine={false} />
                      <YAxis tickFormatter={(value) => money(Number(value))} tick={{ fontSize: 9, fill: "#777e78" }} axisLine={false} tickLine={false} />
                      <Tooltip formatter={(value) => [fullNumber(Number(value)), "Average sales"]} />
                      <Bar dataKey="average" radius={[4, 4, 0, 0]}>
                        {weekdayData.map((item, index) => (
                          <Cell key={item.day} fill={index === 0 ? "#ed762d" : "#f6a15f"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="panel small-chart-panel">
                <div className="small-panel-heading">
                  <div className="small-heading-icon green-text">
                    <TrendingUp size={18} />
                  </div>
                  <div>
                    <h3>Monthly Sales Pattern</h3>
                    <p>Average sales per recorded day</p>
                  </div>
                </div>
                <div className="small-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={monthlyData} margin={{ top: 12, right: 2, left: -20, bottom: 0 }}>
                      <CartesianGrid vertical={false} stroke="#ece8df" />
                      <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#777e78" }} axisLine={false} tickLine={false} />
                      <YAxis tickFormatter={(value) => money(Number(value))} tick={{ fontSize: 9, fill: "#777e78" }} axisLine={false} tickLine={false} />
                      <Tooltip formatter={(value) => [fullNumber(Number(value)), "Average sales"]} />
                      <Bar dataKey="average" fill="#a6c8a8" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article id="performance" className="panel performance-panel">
                <div className="small-panel-heading">
                  <div className="small-heading-icon trophy-icon">
                    <Trophy size={18} />
                  </div>
                  <div>
                    <h3>Model Performance</h3>
                    <p>Rolling evaluation · WAPE</p>
                  </div>
                </div>
                <div className="performance-content">
                  {metrics.length ? (
                    metrics
                      .map((metric) => ({
                        ...metric,
                        model: String(metric.model ?? (metric as Record<string, unknown>)["Model"] ?? ""),
                      }))
                      .filter((metric) => metric.model.trim().length > 0)
                      .map((metric, index) => {
                        const wape = bestWape(metric);
                      return (
                        <div className="performance-row" key={metric.model}>
                          <span className="model-name">{metric.model}</span>
                          <div className="performance-track">
                            <span
                              className={index === 0 ? "baseline-fill" : "model-fill"}
                              style={{
                                width: `${Math.min(100, Math.max(5, wape * 5))}%`,
                              }}
                            />
                          </div>
                          <strong>{wape.toFixed(2)}%</strong>
                        </div>
                      );
                    })
                  ) : (
                    <div className="metrics-empty">
                      <Info size={15} />
                      Model comparison appears when the metrics endpoint is connected.
                    </div>
                  )}
                </div>
                <div className="performance-note">
                  <Info size={14} />
                  Lower WAPE indicates lower weighted forecast error.
                </div>
              </article>
            </section>
          </div>

          <aside id="forecast" className="forecast-column">
            <section className="panel forecast-builder">
              <div className="forecast-title-row">
                <div className="forecast-main-icon">
                  <TrendingUp size={24} />
                </div>
                <div>
                  <h2>Generate Forecast</h2>
                  <p>Estimate daily sales for your selected forecast horizon.</p>
                </div>
              </div>

              <div className="forecast-divider" />

              <div className="horizon-label">
                <span>Forecast Horizon</span>
                <span className="horizon-max">UP TO 7 DAYS</span>
              </div>

              <div className="horizon-options">
                {[1, 3, 5, 7].map((days) => (
                  <button
                    key={days}
                    className={horizon === days ? "chosen" : ""}
                    onClick={() => setHorizon(days)}
                  >
                    {days} {days === 1 ? "day" : "days"}
                  </button>
                ))}
              </div>

              <button
                className="generate-button"
                onClick={generateForecast}
                disabled={loading}
              >
                {loading ? (
                  <LoaderCircle className="spin" size={19} />
                ) : (
                  <Play size={19} fill="currentColor" />
                )}
                {loading ? "Generating…" : "Generate Forecast"}
              </button>

              <div className="builder-footnote">
                <CheckCircle2 size={15} />
                Uses your locally hosted model
              </div>
            </section>

            <section className="panel forecast-results">
              <div className="results-heading">
                <div className="results-icon">
                  <CalendarDays size={21} />
                </div>
                <div>
                  <h2>{forecast.length ? `${forecast.length}-Day Forecast` : "Your Forecast"}</h2>
                  <p>Predicted daily sales</p>
                </div>
                {forecast.length > 0 && (
                  <button
                    className="download-button"
                    onClick={downloadForecast}
                    title="Download forecast CSV"
                    aria-label="Download forecast CSV"
                  >
                    <Download size={17} />
                  </button>
                )}
              </div>

              {forecast.length ? (
                <>
                  <div className="forecast-summary">
                    <span>Predicted total</span>
                    <strong>{fullNumber(forecastTotal)}</strong>
                    <span>Average / day: {fullNumber(forecastAverage)}</span>
                  </div>
                  <div className="forecast-table-wrap">
                    <table className="forecast-table">
                      <thead>
                        <tr>
                          <th>DATE</th>
                          <th>PREDICTED SALES</th>
                        </tr>
                      </thead>
                      <tbody>
                        {forecast.map((row) => (
                          <tr key={row.date}>
                            <td>
                              <span className="table-date-dot" />
                              {row.date}
                            </td>
                            <td>{fullNumber(row.predicted_sales)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="forecast-placeholder">
                  <div className="placeholder-graphic">
                    <LineChart size={34} />
                  </div>
                  <strong>Predictions appear here</strong>
                  <p>Choose a horizon above and generate your first forecast.</p>
                </div>
              )}
            </section>

            <section className="note-card">
              <div className="note-icon">
                <Info size={19} />
              </div>
              <div>
                <strong>Forecasting note</strong>
                <p>
                  This project uses historical data ending in August 2017.
                  Future holidays and business conditions may differ, so
                  validate forecasts before making operational decisions.
                </p>
              </div>
            </section>

            <div className="dataset-footer">
              <Clock3 size={14} />
              <span>HISTORICAL DATASET · 2013–2017</span>
              <ChevronDown size={14} />
            </div>
          </aside>
        </section>

        <footer className="page-footer">
          <span>NEXUS<span className="brand-dot">.</span> RETAIL INTELLIGENCE</span>
          <span>Built with Python · FastAPI · React</span>
        </footer>
      </main>
    </div>
  );
}

export default App;