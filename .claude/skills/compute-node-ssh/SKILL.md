# Compute Node SSH Port Forwarding (O2 Cluster)

How to forward a port from an O2 compute node to your local machine for web apps (Dash, Jupyter, etc.).

## Requirements

1. **The app must bind to `0.0.0.0`**, not `127.0.0.1` (the default for most frameworks). Otherwise the SSH tunnel can connect but gets "connection refused".
   - Dash: `app.run(host="0.0.0.0", port=PORT)`
   - Jupyter: `--ip=0.0.0.0`

2. **Use a specific login node, not the load balancer.** The load balancer (`o2.hms.harvard.edu`) doesn't reliably forward ports. Find which login node you're connected through:
   ```bash
   # On the compute node:
   echo $SSH_CONNECTION
   # Take the second IP and reverse-lookup:
   host <that_ip>
   # e.g. → login03.o2.rc.hms.harvard.edu
   ```

3. **SSH command (run from your local machine):**
   ```bash
   ssh -L <PORT>:<COMPUTE_NODE>:<PORT> -J joj144@<LOGIN_NODE>.o2.rc.hms.harvard.edu joj144@<COMPUTE_NODE>
   ```
   Example:
   ```bash
   ssh -L 8050:compute-g-17-165:8050 -J joj144@login03.o2.rc.hms.harvard.edu joj144@compute-g-17-165
   ```

4. Then open `http://localhost:<PORT>` in your local browser.

## Notes

- The tunnel stays open as long as the SSH session is connected — no need to reconnect if you restart the app on the same port.
- Get the compute node hostname with `hostname` on the compute node.
- If you're on Claude Code, launch the app in the background with `run_in_background` and give the user the SSH command.
