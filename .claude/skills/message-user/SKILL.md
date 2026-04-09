---
name: message-user
description: Use when you need to send the user a notification, alert, or message outside of the terminal — e.g., when a long-running task completes, something needs attention, or the user asked to be pinged.
---

# Message User

Send a Discord notification to the user via webhook.

```bash
curl -s -H "Content-Type: application/json" \
  -d '{"content": "<message>"}' \
  "https://discord.com/api/webhooks/1491608945591058493/kF7qp7i3fAGtYz-dYQjgZhUlZsWxAFT62yEWXHRL0Sd-9btiyMx9EWx6CtJ_1nPlDnwE"
```

Replace `<message>` with the notification text. Keep messages concise. This posts to the user's Discord server and notifies on both desktop and phone.
