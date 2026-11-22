---
name: interview-prep
description: Prepare the user for an upcoming job interview with company and role research, past emails, likely questions and talking points.
---
1. Find the interview: the one the user named, or the next interview on the calendar
   (`list_calendar_events` with days_ahead=7; look for "interview", "screen", "chat with").
   If there's none and you're in a chat, ask which one. In a background task, say so and stop.
2. Gather what's known:
   - `list_job_applications` for the company and role, and its thread
   - `read_email_thread` for that thread: role details, who they'll meet, format, links
   - `search_memory` for the user's background: "experience", "projects", "skills"
3. Research (only facts you found; name the source of each):
   - `web_search` "<company> what they do", "<company> news", and the job posting if
     you can find it; `read_web_page` on the 2-3 best results
4. Write the prep, easy to read on a phone:
   - Interview: when, with whom, format and link
   - The company in 5 bullets: product, customers, size, recent news, tech if known
   - The role: what they're likely looking for
   - 6 likely questions (technical and behavioural), each with a one-line angle from the
     user's own projects and experience
   - 3 good questions for the user to ask them
   - Checklist: link tested, CV open, notes ready, water
5. Don't save the prep to memory: it's for this one interview.
