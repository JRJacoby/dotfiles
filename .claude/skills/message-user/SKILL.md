---
name: message-user
description: Use when you need to send the user a notification, alert, or message outside of the terminal — e.g., when a long-running task completes, something needs attention, or the user asked to be pinged. Sends via Discord (default), email, or Slack.
---

# Message User

Send the user a notification outside the terminal. **Default to Discord** unless the user explicitly asks for Slack.

## NEVER decide the user doesn't need the ping

**If the user asked to be notified, send the notification. Every time. No exceptions.**

You cannot observe whether the user is at their computer, watching the terminal,
or reading your output. You have no signal for this and never will. A message
from them in the conversation is not evidence they are present *now* — they may
have typed it and walked away, and terminal output is not a notification.

Do not suppress, skip, defer, or downgrade a requested notification because of
any of the following reasoning. Each one is invalid:

- "They're active in the conversation, so they already know."
- "I already reported this in my last message, a ping would be redundant."
- "They're clearly at their desk right now."
- "This is only a small update / a failure they're expecting."
- "The notification was set up assuming they'd be away, and they're not."
- "Sending it would be noise."

Redundancy is not your call to make. A duplicate notification costs the user
nothing; a missing one means they sit waiting on a job that already finished, or
assume something is broken because the ping never arrived. That failure is
silent and expensive, and the user cannot tell the difference between "nothing
happened yet" and "Claude decided not to tell me."

The **only** reason not to send a requested notification is that the user
explicitly told you to stop sending them. If you think a notification is
genuinely unnecessary, send it anyway and *then* ask whether they want the
cadence changed.

This applies to every notification path in this skill, and to notifications
requested through any other mechanism — a `/loop`, a cron job, a monitor, or a
plain "ping me when X".

## Discord (default)

```bash
curl -s -H "Content-Type: application/json" \
  -d '{"content": "<message>"}' \
  "https://discord.com/api/webhooks/1491608945591058493/kF7qp7i3fAGtYz-dYQjgZhUlZsWxAFT62yEWXHRL0Sd-9btiyMx9EWx6CtJ_1nPlDnwE"
```

Replace `<message>` with the notification text. Keep it concise. Posts to the user's Discord server and notifies on desktop + phone.

### Attaching a file (image, video, plot, etc.)

The webhook accepts file uploads via multipart form: send `file=@<path>` and put the text in a `payload_json` part instead of `-d`. Images and H.264/yuv420p mp4s render/play inline; other types show as a download.

```bash
curl -s -w '\nHTTP %{http_code}' \
  -F 'payload_json={"content":"<message>"}' \
  -F "file=@<path>;type=<mime>" \
  "https://discord.com/api/webhooks/1491608945591058493/kF7qp7i3fAGtYz-dYQjgZhUlZsWxAFT62yEWXHRL0Sd-9btiyMx9EWx6CtJ_1nPlDnwE"
```

- Set `type=` to the real MIME (`video/mp4`, `image/png`, …) so Discord previews it inline.
- **Size cap ~8 MB** for the standard webhook; larger files return HTTP 413. Shrink first (re-encode/downscale) or upload elsewhere and post the link.
- A successful upload returns HTTP 200 with an `attachments` array echoing the filename/size — check for that, not just exit code.
- To attach multiple files, repeat `-F "file1=@..."`, `-F "file2=@..."`.

## Email

For longer messages, or anything the user wants to read in their inbox, email `john_jacoby@hms.harvard.edu`. From an O2 **login node**, `mail` works and lands in the Inbox within seconds (tested from login01; it did not go to Junk):

```bash
echo "<message body>" | mail -s "<subject>" john_jacoby@hms.harvard.edu
```

From a compute node, `mail` hangs: `postdrop` cannot write the postfix maildrop spool (`Permission denied`) and retries forever. Either run it on a login node (`ssh -o BatchMode=yes login.o2.rc.hms.harvard.edu '... | mail ...'`) or submit over SMTP to `localhost:25` as in the Slack section below, with the user's address as `To`. Wrap a `mail` call in `timeout` so a node with a broken spool fails instead of hanging.

The Gmail and Microsoft 365 connectors can only create drafts, not send, so they are no use for notifications.

## Slack (when requested)

Email the user's Slack channel via its "Send emails to this channel" address. The email **Subject becomes the bold message title** in Slack and the **body becomes the content**; email **attachments post as files** in the channel.

On O2 the `mail`/`sendmail` command fails on some nodes (`postdrop: ... Permission denied` from the postfix maildrop spool), so submit directly over SMTP to `localhost:25`:

```python
import smtplib, socket
from email.message import EmailMessage

msg = EmailMessage()
msg["From"] = f"joj144@{socket.getfqdn()}"
msg["To"] = "test-aaaauqnmx5rrivw4mingxsov6y@harvard.org.slack.com"  # user's Slack channel email
msg["Subject"] = "<short title>"     # shown as the bold message heading in Slack
msg.set_content("<message body>")
# optional: attach a file (shows up as a file in the channel)
# data = open("plot.png", "rb").read()
# msg.add_attachment(data, maintype="image", subtype="png", filename="plot.png")
with smtplib.SMTP("localhost", 25, timeout=60) as s:
    s.send_message(msg)
```

Limits: **channel-only** — the address is per-channel, so this cannot search Slack or post to arbitrary channels/DMs/group chats (that needs an admin-approved bot app). The address above is the **test channel**; swap in a dedicated channel address if the user sets one up.
