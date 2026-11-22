---
name: follow-up-nudges
description: Find job applications and recruiter conversations that have gone quiet and draft short, polite follow-up emails for the user to send.
schedule: mon 09:00
---
1. `list_job_applications`. Candidates are applications in stage applied, screening or
   interview whose last contact is at least 7 days ago (10 days for "applied").
   Never follow up on rejected, withdrawn or offer.
2. For each candidate (at most 3 per run, oldest first), `read_email_thread` on its
   thread. Skip it if the last message is from the user and is less than 7 days old, or if
   the other side said when they'd reply and that date hasn't passed.
3. Draft the follow-up with `draft_email_reply`, passing `reply_to_thread_id` so it stays in
   the thread (and so Send is offered for people already in it):
   - 3 to 4 sentences, warm and brief: thank them, restate interest in the role by name,
     ask about next steps or timing.
   - Use only facts from the thread. Never invent dates, names or details.
4. The user chooses Send, Save as draft or Reject. After a Send or a saved draft, update
   the tracker: `save_job_application` with next_step "Followed up on <date>".
5. Finish with: who you drafted for, and who is still waiting but not due yet.
