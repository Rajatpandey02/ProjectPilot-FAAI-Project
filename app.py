from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


from projectpilot import availability as av
from projectpilot import task_store as ts
from projectpilot.config import (
    REQUIRED_IMPORT_COLUMNS, STATUSES, TASK_TYPES, TASKS_FILE, WEEKDAYS,
)
from projectpilot.synthetic import generate_synthetic_tasks
from projectpilot.agents import PriorityAgent, SchedulingAgent
from projectpilot.tools import default_registry
st.set_page_config(page_title="ProjectPilot Campus", page_icon="🎓", layout="wide")


def init_state() -> None:
    if "tasks" not in st.session_state:
        if TASKS_FILE.exists():
            st.session_state.tasks = ts.load_tasks()
        else:  # first run: seed demo data so the app is not empty
            st.session_state.tasks = generate_synthetic_tasks()
            ts.save_tasks(st.session_state.tasks)
    if "availability" not in st.session_state:
        st.session_state.availability = av.load_availability()


def commit(df: pd.DataFrame) -> None:
    st.session_state.tasks = df
    ts.save_tasks(df)


init_state()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 ProjectPilot Campus")
    st.caption("Step 1: Task manager")
    if st.button("Load synthetic demo data", use_container_width=True):
        commit(generate_synthetic_tasks())
    if st.button("Clear all tasks", use_container_width=True):
        commit(ts.normalize(pd.DataFrame()))
    st.download_button(
        "Download tasks as CSV",
        ts.tasks_to_csv_bytes(st.session_state.tasks),
        file_name="tasks_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

tab_dash, tab_prio, tab_sched, tab_add, tab_import, tab_manage, tab_avail = st.tabs(
    ["📊 Dashboard", "🎯 Priorities", "📅 Schedule", "➕ Add task", "📥 Import CSV",
     "✏️ Manage tasks", "⏱️ Study hours"]
)

# -------------------------------------------------------------- dashboard
with tab_dash:
    today = pd.Timestamp(date.today())
    data = ts.enrich(st.session_state.tasks, today)
    active = data[data["status"] != "Completed"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tasks", len(data))
    c2.metric("Active tasks", len(active))
    c3.metric("Overdue", int(active["is_overdue"].sum()))
    c4.metric("Remaining effort (h)", f"{active['remaining_hours'].sum():.1f}")

    if active.empty:
        st.info("No active tasks. Add one, import a CSV, or load the demo data from the sidebar.")
    else:
        week_end = today + pd.Timedelta(days=6)
        needed = active.loc[active["deadline"] <= week_end, "remaining_hours"].sum()
        capacity = av.capacity_between(st.session_state.availability, today, week_end)
        if needed > capacity:
            st.warning(
                f"Next 7 days: {needed:.1f} h of work due vs {capacity:.1f} h available "
                f"(shortfall {needed - capacity:.1f} h). The Scheduling and Recovery agents will handle this in later steps."
            )
        else:
            st.success(f"Next 7 days: {needed:.1f} h of work due vs {capacity:.1f} h available.")

        left, right = st.columns(2)
        by_course = active.groupby("course", as_index=False)["remaining_hours"].sum()
        left.plotly_chart(
            px.bar(by_course, x="course", y="remaining_hours", title="Remaining effort by course (h)"),
            use_container_width=True,
        )
        upcoming = active.sort_values("deadline")
        fig = px.bar(
            upcoming, x="days_left", y="title", color="status", orientation="h",
            hover_data=["course", "task_type", "remaining_hours"],
            title="Days until deadline (negative = overdue)",
        )
        fig.update_yaxes(autorange="reversed", title=None)
        right.plotly_chart(fig, use_container_width=True)

    st.subheader("All tasks by deadline")
    st.dataframe(
        data.sort_values("deadline")[
            ["task_id", "title", "course", "task_type", "deadline", "estimated_hours",
             "hours_completed", "remaining_hours", "days_left", "status"]
        ],
        hide_index=True,
        use_container_width=True,
    )

# ------------------------------------------------------------- priorities
TIER_COLORS = {"Critical": "#d62728", "High": "#ff7f0e", "Medium": "#f2c037", "Low": "#2ca02c"}

with tab_prio:
    st.caption("The Priority Agent scores each active task with tools, applies decision rules, "
               "and re-ranks when your tasks or study hours change.")
    if st.session_state.tasks.empty:
        st.info("No tasks yet. Add some or load the demo data from the sidebar.")
    else:
        result = PriorityAgent().run(st.session_state.tasks, st.session_state.availability)
        table = result.table

        cols = st.columns(5)
        for col, tier in zip(cols[:4], ["Critical", "High", "Medium", "Low"]):
            col.metric(tier, int((table["priority_tier"] == tier).sum()))
        cols[4].metric("At risk", int(table["at_risk"].sum()))

        if not table.empty:
            fig = px.bar(
                table, x="priority_score", y="title", color="priority_tier",
                orientation="h", color_discrete_map=TIER_COLORS,
                title="Priority score by task (bars are in rank order)",
                hover_data=["course", "deadline", "reasons"],
            )
            fig.update_yaxes(autorange="reversed", title=None)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                table[["priority_rank", "title", "course", "task_type", "deadline", "days_left",
                       "remaining_hours", "capacity_hours", "pressure_ratio",
                       "priority_score", "priority_tier", "at_risk", "reasons"]],
                hide_index=True,
                use_container_width=True,
                column_config={"deadline": st.column_config.DateColumn(format="YYYY-MM-DD")},
            )

        with st.expander("🧠 Agent decision trace"):
            for step in result.trace:
                st.write(f"`{step.kind}` {step.message}")
            st.caption("Tool calls this run: " + ", ".join(f"{k} ×{v}" for k, v in result.tool_calls.items()))
        with st.expander("🔧 Tools available to agents"):
            for name, desc in default_registry.describe():
                st.write(f"**{name}**: {desc}")

# --------------------------------------------------------------- schedule
STATUS_COLORS = {"On track": "#2ca02c", "Late": "#d62728",
                 "Overdue (catch-up)": "#ff7f0e", "Unscheduled": "#7f7f7f"}

with tab_sched:
    st.caption("The Scheduling Agent builds an earliest-deadline-first plan inside your study hours, "
               "defers the lowest-priority work when deadlines cannot all be met, and validates the result.")
    if st.session_state.tasks.empty:
        st.info("No tasks yet. Add some or load the demo data from the sidebar.")
    else:
        max_session = st.slider("Longest single study session (hours)", 1.0, 4.0, 2.0, 0.5)
        prio = PriorityAgent().run(st.session_state.tasks, st.session_state.availability)
        plan = SchedulingAgent().run(prio.table, st.session_state.availability,
                                     max_session_hours=max_session)
        status_counts = plan.task_plan["plan_status"].value_counts()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Planned hours", f"{plan.schedule['hours'].sum():.1f}")
        m2.metric("On track", int(status_counts.get("On track", 0)))
        m3.metric("Late / overdue / unscheduled", int(len(plan.task_plan) - status_counts.get("On track", 0)))
        m4.metric("High-severity conflicts", int((plan.conflicts["severity"] == "High").sum()))

        if plan.schedule.empty:
            st.info("Nothing to schedule. Every active task is complete or availability is zero.")
        else:
            st.subheader("Today")
            todays = plan.schedule[plan.schedule["date"] == pd.Timestamp(date.today())]
            if todays.empty:
                st.write("Nothing scheduled today.")
            for r in todays.itertuples():
                st.write(f"- **{r.hours:g}h** {r.title} ({r.course}, session {r.session})")

            st.subheader("Conflicts and risks")
            if plan.conflicts.empty:
                st.success("No conflicts detected.")
            for r in plan.conflicts.itertuples():
                text = f"**{r.type}**: {r.message}"
                {"High": st.error, "Medium": st.warning}.get(r.severity, st.info)(text)

            gantt = plan.task_plan.dropna(subset=["start_date"]).copy()
            gantt["end"] = gantt["finish_date"] + pd.Timedelta(days=1)
            fig = px.timeline(
                gantt, x_start="start_date", x_end="end", y="title", color="plan_status",
                color_discrete_map=STATUS_COLORS, hover_data=["course", "scheduled_hours", "buffer_days"],
                title="Work window per task (◆ = deadline)",
            )
            fig.add_trace(go.Scatter(x=gantt["deadline"], y=gantt["title"], mode="markers",
                                     marker=dict(symbol="diamond", size=11, color="black"), name="Deadline"))
            fig.update_yaxes(autorange="reversed", title=None)
            st.plotly_chart(fig, use_container_width=True)

            by_course = plan.schedule.groupby(["date", "course"], as_index=False)["hours"].sum()
            fig2 = px.bar(by_course, x="date", y="hours", color="course",
                          title="Scheduled hours per day vs available hours")
            fig2.add_trace(go.Scatter(x=plan.daily_load["date"], y=plan.daily_load["capacity"],
                                      mode="lines+markers", name="Available",
                                      line=dict(color="black", dash="dot")))
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Day-by-day plan")
            st.dataframe(
                plan.schedule[["date", "weekday", "slot", "title", "course", "task_type", "hours",
                               "session", "deadline", "priority_tier", "plan_status"]],
                hide_index=True, use_container_width=True,
                column_config={"date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                               "deadline": st.column_config.DateColumn(format="YYYY-MM-DD")},
            )
            st.download_button("Download plan as CSV", plan.schedule.to_csv(index=False).encode("utf-8"),
                               file_name="study_plan.csv", mime="text/csv")

            with st.expander("Per-task plan"):
                st.dataframe(plan.task_plan, hide_index=True, use_container_width=True)

        with st.expander("🧠 Scheduling Agent decision trace"):
            for step in plan.trace:
                st.write(f"`{step.kind}` {step.message}")
            st.caption("Tool calls this run: " + ", ".join(f"{k} ×{v}" for k, v in plan.tool_calls.items()))
            
# --------------------------------------------------------------- add task
with tab_add:
    with st.form("add_task_form", clear_on_submit=True):
        title = st.text_input("Title")
        a, b = st.columns(2)
        course = a.text_input("Course")
        task_type = b.selectbox("Type", TASK_TYPES)
        deadline = a.date_input("Deadline", value=date.today() + timedelta(days=7))
        est = b.number_input("Estimated effort (hours)", min_value=0.5, value=3.0, step=0.5)
        done = a.number_input("Hours already completed", min_value=0.0, value=0.0, step=0.5)
        status = b.selectbox("Status", STATUSES)
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Add task")
    if submitted:
        new_df, errors = ts.add_task(
            st.session_state.tasks, title, course, task_type, deadline, est, done, status, notes
        )
        if errors:
            st.error("\n".join(errors))
        else:
            commit(new_df)
            st.success(f"Added '{title}'.")

# ------------------------------------------------------------- import CSV
with tab_import:
    st.write(f"Required columns: `{', '.join(REQUIRED_IMPORT_COLUMNS)}`. "
             "Optional: `task_id, hours_completed, status, notes`. Dates as YYYY-MM-DD.")
    st.download_button(
        "Download a sample CSV to test with",
        ts.tasks_to_csv_bytes(generate_synthetic_tasks()),
        file_name="sample_tasks.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    mode = st.radio("Import mode", ["Append to existing tasks", "Replace all tasks"], horizontal=True)

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            raw.columns = [c.strip().lower() for c in raw.columns]
        except Exception as exc:  # unreadable or malformed file
            st.error(f"Could not read the CSV: {exc}")
            raw = None

        if raw is not None:
            missing = [c for c in REQUIRED_IMPORT_COLUMNS if c not in raw.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                clean, errors = ts.validate_tasks(raw)
                st.write(f"**{len(clean)}** valid row(s), **{len(errors)}** problem(s).")
                if errors:
                    with st.expander("Problems found (these rows will be skipped)"):
                        for e in errors:
                            st.write("•", e)
                st.dataframe(clean, hide_index=True, use_container_width=True)
                if st.button("Confirm import", disabled=clean.empty):
                    if mode.startswith("Append"):
                        incoming = clean.assign(task_id="")  # regenerate IDs to avoid clashes
                        existing = st.session_state.tasks
                        combined = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
                        merged, _ = ts.validate_tasks(combined)
                    else:
                        merged = clean
                    commit(merged)
                    st.success(f"Imported {len(clean)} task(s). Total now: {len(merged)}.")

# ----------------------------------------------------------- manage tasks
with tab_manage:
    st.caption("Edit cells directly, change status, log hours, add or delete rows, then click Save.")
    edited = st.data_editor(
        st.session_state.tasks,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="task_editor",
        column_config={
            "task_id": st.column_config.TextColumn("ID", disabled=True),
            "deadline": st.column_config.DateColumn("Deadline", format="YYYY-MM-DD"),
            "task_type": st.column_config.SelectboxColumn("Type", options=TASK_TYPES),
            "status": st.column_config.SelectboxColumn("Status", options=STATUSES),
            "estimated_hours": st.column_config.NumberColumn("Est. hours", min_value=0.5, step=0.5),
            "hours_completed": st.column_config.NumberColumn("Done (h)", min_value=0.0, step=0.5),
        },
    )
    if st.button("Save changes"):
        clean, errors = ts.validate_tasks(edited)
        if errors:
            st.error("Not saved. Fix these first:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            commit(clean)
            st.success("Saved.")

# ------------------------------------------------------------ availability
with tab_avail:
    st.write("How many hours can you study on each weekday?")
    cols = st.columns(7)
    new_avail = {
        day: col.number_input(
            day[:3], min_value=0.0, max_value=16.0, step=0.5,
            value=float(st.session_state.availability[day]), key=f"avail_{day}",
        )
        for col, day in zip(cols, WEEKDAYS)
    }
    st.metric("Weekly total (h)", f"{sum(new_avail.values()):.1f}")
    if st.button("Save availability"):
        st.session_state.availability = new_avail
        av.save_availability(new_avail)
        st.success("Availability saved.")