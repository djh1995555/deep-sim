import html
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DEBUG_PANELS: Dict[str, Sequence[str]] = {
    "Speed Tracking": (
        "input.vx",
        "vehicle.vx",
        "input.target_speed_mps",
        "output.debug.longitudinal.speed_error",
        "output.debug.longitudinal.accel_cmd",
    ),
    "Lateral Tracking": (
        "input.y_world",
        "input.target_y_m",
        "output.debug.lateral.lateral_error_m",
        "input.yaw",
        "input.target_yaw_rad",
        "output.debug.lateral.heading_error_rad",
    ),
    "Commands": (
        "output.sw_angle",
        "output.steer_cmd",
        "output.throttle_cmd",
        "output.brake_cmd",
        "output.debug.lateral.raw_sw_angle",
    ),
    "Vehicle State": (
        "vehicle.vy",
        "vehicle.r",
        "vehicle.mu_min",
        "vehicle.friction_usage_max",
    ),
    "Reference": (
        "input.target_x_m",
        "input.path_s_m",
        "input.lookahead_distance_m",
        "input.target_curvature_1pm",
        "input.target_yaw_rate_rps",
    ),
}


def write_plotly_debug_report(
    rows: Sequence[Dict[str, Any]],
    path: str,
    panels: Optional[Dict[str, Sequence[str]]] = None,
    title: str = "Simulator Debug Report",
) -> None:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError("plotly is required to write debug_report.html") from exc

    panel_defs = panels or DEFAULT_DEBUG_PANELS
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    figures = []
    trajectory = _build_trajectory_figure(rows)
    if trajectory is not None:
        figures.append(("Trajectory", trajectory))
    timeseries = _build_timeseries_figure(rows, panel_defs)
    if timeseries is not None:
        figures.append(("Timeseries", timeseries))

    body = "\n".join(_figure_card(name, figure) for name, figure in figures)
    if not body:
        body = "<p>No numeric debug signals were available.</p>"
    document = _html_document(title, get_plotlyjs(), body, _summary_table(rows))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def _build_timeseries_figure(
    rows: Sequence[Dict[str, Any]],
    panels: Dict[str, Sequence[str]],
) -> Optional[Any]:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    active_panels: List[Tuple[str, List[Tuple[str, List[float], List[float]]]]] = []
    for panel_name, signals in panels.items():
        traces = []
        for signal in signals:
            x_values, y_values = _numeric_series(rows, signal)
            if _has_valid_value(y_values):
                traces.append((signal, x_values, y_values))
        if traces:
            active_panels.append((panel_name, traces))
    if not active_panels:
        return None

    fig = make_subplots(
        rows=len(active_panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=[name for name, _ in active_panels],
    )
    for row_index, (_, traces) in enumerate(active_panels, start=1):
        for signal, x_values, y_values in traces:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=signal,
                    connectgaps=False,
                ),
                row=row_index,
                col=1,
            )
        fig.update_yaxes(title_text="", row=row_index, col=1)
    fig.update_xaxes(title_text="time [s]", row=len(active_panels), col=1)
    fig.update_layout(
        template="plotly_dark",
        height=max(320, 260 * len(active_panels)),
        margin={"l": 60, "r": 260, "t": 70, "b": 45},
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        hovermode="x unified",
    )
    return fig


def _build_trajectory_figure(rows: Sequence[Dict[str, Any]]) -> Optional[Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    vehicle_x, vehicle_y = _xy_series(rows, "vehicle.x_world", "vehicle.y_world")
    time_values = [float(row.get("t", index)) for index, row in enumerate(rows)]
    if _has_valid_value(vehicle_x) and _has_valid_value(vehicle_y):
        fig.add_trace(
            go.Scatter(
                x=vehicle_x,
                y=vehicle_y,
                mode="lines",
                name="actual vehicle trajectory",
                customdata=time_values,
                hovertemplate=(
                    "actual<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>"
                    "t=%{customdata:.3f} s<extra></extra>"
                ),
            )
        )
    reference_x, reference_y = _xy_series(rows, "input.target_x_m", "input.target_y_m")
    if _has_valid_value(reference_x) and _has_valid_value(reference_y):
        fig.add_trace(
            go.Scatter(
                x=reference_x,
                y=reference_y,
                mode="lines",
                name="reference trajectory",
                customdata=time_values,
                hovertemplate=(
                    "reference<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>"
                    "t=%{customdata:.3f} s<extra></extra>"
                ),
            )
        )
    if not fig.data:
        return None
    _add_endpoint_markers(fig, rows)
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin={"l": 60, "r": 220, "t": 55, "b": 55},
        xaxis_title="x_world [m]",
        yaxis_title="y_world [m]",
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        hovermode="closest",
        dragmode="pan",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return fig


def _add_endpoint_markers(fig: Any, rows: Sequence[Dict[str, Any]]) -> None:
    import plotly.graph_objects as go

    endpoints = [
        ("actual start", "vehicle.x_world", "vehicle.y_world", 0, "#22c55e"),
        ("actual end", "vehicle.x_world", "vehicle.y_world", -1, "#ef4444"),
        ("reference start", "input.target_x_m", "input.target_y_m", 0, "#84cc16"),
        ("reference end", "input.target_x_m", "input.target_y_m", -1, "#f97316"),
    ]
    for name, x_key, y_key, index, color in endpoints:
        if not rows:
            continue
        row = rows[index]
        x_value = _numeric_or_nan(row.get(x_key))
        y_value = _numeric_or_nan(row.get(y_key))
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[y_value],
                mode="markers",
                name=name,
                marker={"size": 10, "color": color, "symbol": "circle"},
                hovertemplate="%{fullData.name}<br>x=%{x:.3f} m<br>y=%{y:.3f} m<extra></extra>",
            )
        )


def _numeric_series(
    rows: Sequence[Dict[str, Any]],
    key: str,
) -> Tuple[List[float], List[float]]:
    x_values = [float(row.get("t", index)) for index, row in enumerate(rows)]
    y_values = [_numeric_or_nan(row.get(key)) for row in rows]
    return x_values, y_values


def _xy_series(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
) -> Tuple[List[float], List[float]]:
    return (
        [_numeric_or_nan(row.get(x_key)) for row in rows],
        [_numeric_or_nan(row.get(y_key)) for row in rows],
    )


def _numeric_or_nan(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    return float("nan")


def _has_valid_value(values: Iterable[float]) -> bool:
    return any(math.isfinite(value) for value in values)


def _summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    duration = 0.0
    if rows:
        duration = float(rows[-1].get("t", 0.0)) - float(rows[0].get("t", 0.0))
    items = [
        ("samples", str(len(rows))),
        ("duration_s", "%.3f" % duration),
    ]
    cells = "".join(
        "<tr><th>%s</th><td>%s</td></tr>" % (html.escape(key), html.escape(value))
        for key, value in items
    )
    return "<table class=\"summary\"><tbody>%s</tbody></table>" % cells


def _figure_card(name: str, figure: Any) -> str:
    figure_html = figure.to_html(include_plotlyjs=False, full_html=False)
    return (
        "<section class=\"figure-card\">"
        "<h2>%s</h2>"
        "%s"
        "</section>"
    ) % (html.escape(name), figure_html)


def _html_document(title: str, plotlyjs: str, body: str, summary: str) -> str:
    safe_title = html.escape(title)
    return """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script type="text/javascript">{plotlyjs}</script>
    <style>
      body {{
        margin: 0;
        background: #0f1117;
        color: #e5e7eb;
        font-family: Arial, Helvetica, sans-serif;
      }}
      header {{
        padding: 24px 32px 12px;
      }}
      h1 {{
        margin: 0 0 16px;
        font-size: 24px;
        font-weight: 700;
      }}
      h2 {{
        margin: 0 0 12px;
        font-size: 18px;
        font-weight: 700;
      }}
      .summary {{
        border-collapse: collapse;
        color: #d1d5db;
      }}
      .summary th,
      .summary td {{
        border: 1px solid #374151;
        padding: 6px 10px;
        text-align: left;
      }}
      .figure-card {{
        margin: 16px 24px 28px;
        padding: 16px;
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
      }}
    </style>
  </head>
  <body>
    <header>
      <h1>{title}</h1>
      {summary}
    </header>
    {body}
  </body>
</html>
""".format(title=safe_title, plotlyjs=plotlyjs, summary=summary, body=body)
