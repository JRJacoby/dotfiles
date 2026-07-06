---
name: message-user
description: Use when you need to send the user a notification, alert, or message outside of the terminal — e.g., when a long-running task completes, something needs attention, or the user asked to be pinged. Sends via Discord (default) or Slack.
---

# Message User

Send the user a notification outside the terminal. **Default to Discord** unless the user explicitly asks for Slack.

## Discord (default)

```bash
curl -s -H "Content-Type: application/json" \
  -d '{"content": "<message>"}' \
  "https://discord.com/api/webhooks/1491608945591058493/kF7qp7i3fAGtYz-dYQjgZhUlZsWxAFT62yEWXHRL0Sd-9btiyMx9EWx6CtJ_1nPlDnwE"
```

Replace `<message>` with the notification text. Keep it concise. Posts to the user's Discord server and notifies on desktop + phone.

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
