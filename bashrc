# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias ..='cd ..'
alias uq='squeue -u joj144'
alias run-gpu='sbatch -J main-interactive -n 1 -c 4 --mem 24G -t 24:00:00 -p gpu_quad --qos=gpuquad_qos --gres=gpu:1 -x "compute-g-17-164" ~/gpu-sbatch.sh'
alias run-ampere-gpu='sbatch -J main-interactive-ampere -n 1 -c 4 --mem 24G -t 24:00:00 -p gpu_quad --qos=gpuquad_qos --gres=gpu:1 --nodes=1 --nodelist="compute-gc-17-[249,252-254,239-240],compute-g-17-[162-163,166-171,200-205]" ~/gpu-sbatch.sh'
alias john='cd /n/groups/datta/john'
alias wq="watch 'squeue -u joj144'"

function ct() {
    rm -f ~/.cursor/cli/*.lock
    rm -f ~/.cursor/cli/servers/*/pid.txt
    rm -f ~/.cursor/cli/servers/*/log.txt
    cursor tunnel --name o2tunnel
}

# --- HISTORY CONFIGURATION ---

# 1. Increase History Size (100k lines)
export HISTSIZE=100000
export HISTFILESIZE=100000

# 2. Append Mode (Don't overwrite the file on exit)
shopt -s histappend

# 3. Smart History Control
# ignoredups: Don't save "ls" twice if run back-to-back
# erasedups: Remove older duplicates to keep the file cleaner
export HISTCONTROL=ignoredups:erasedups

# 4. Immediate Save (Write to disk after every command)
# This ensures that if your SSH session disconnects, you don't lose your history.
# We append to the existing PROMPT_COMMAND so we don't break other settings.
export PROMPT_COMMAND="history -a; $PROMPT_COMMAND"

# OPTIONAL: Instant Read (See commands from other terms immediately)
# WARNING: On network drives (like your cluster) with a 100k file,
# this might make your prompt feel slow/laggy.
# If you want it, uncomment the line below:
# export PROMPT_COMMAND="history -a; history -n; $PROMPT_COMMAND"

export LANGUAGE=UTF-8
export LC_ALL=en_US.UTF-8
export LANG=UTF-8
export LC_CTYPE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_COLLATE=$LANG
export LC_CTYPE=$LANG
export LC_MESSAGES=$LANG
export LC_MONETARY=$LANG
export LC_NUMERIC=$LANG
export LC_TIME=$LANG
export LC_ALL=$LANG
export TMUX_TMPDIR=/n/groups/datta/john/tmux_tmp

export VSCODE_SERVER_DIR=$HOME/.vscode-server

export PATH=$PATH:/home/joj144/local/go/bin
export PATH=$PATH:/home/joj144/go/bin
export PATH=/home/joj144/bin:$PATH

export UV_INDEX_STRATEGY=first-index

function g() {
	gemini -p "$*"
}

function dt() {
	devtunnel host -p "$1" --allow-anonymous
}

# The next line updates PATH for the Google Cloud SDK.
if [ -f '/n/groups/datta/john/local/google-cloud/path.bash.inc' ]; then . '/n/groups/datta/john/local/google-cloud/path.bash.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/n/groups/datta/john/local/google-cloud/completion.bash.inc' ]; then . '/n/groups/datta/john/local/google-cloud/completion.bash.inc'; fi

# Cursor agent: disable conda prompt modifications for clean agent prompts
if [[ ${TERM_PROGRAM:-} == "vscode" || ${TERM_PROGRAM:-} == "cursor" ]]; then
  # Disable conda prompt modifications for Cursor
  export CONDA_CHANGEPS1=no
  export MAMBA_NO_CHANGEPS1=1
  export CONDA_PROMPT_MODIFIER=""
  unset PROMPT_COMMAND
  trap - DEBUG
  export PS1='[\u@\h \W]\$ '
fi

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba shell init' !!
export MAMBA_EXE='/home/joj144/miniforge3/bin/mamba';
export MAMBA_ROOT_PREFIX='/home/joj144/miniforge3';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/joj144/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/joj144/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/joj144/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/joj144/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion


# --- Helper: Get job ID by name from squeue ---
function _get_job_id_by_name() {
    local job_name="$1"
    local squeue_output
    squeue_output=$(squeue -u joj144 --Format="JobID,Name:100" --noheader 2>/dev/null)
    
    if [ -z "$squeue_output" ]; then
        return 1
    fi
    
    # Parse squeue output to find matching job name
    while IFS= read -r line; do
        local line_job_id line_job_name
        line_job_id=$(echo "$line" | awk '{print $1}')
        line_job_name=$(echo "$line" | awk -F'"' '{print $2}')
        
        if [ "$line_job_name" = "$job_name" ]; then
            echo "$line_job_id"
            return 0
        fi
    done <<< "$squeue_output"
    
    return 1
}

# --- (sa) salloc: Allocates a new interactive node ---
function sa() {
    # Default values
    local ampere_gpus=0
    local job_name=""
    local default_regular_name="main-interactive"
    local default_ampere_name="main-interactive-ampere"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ampere|-a)
                if [[ $2 =~ ^[0-9]+$ ]]; then
                    ampere_gpus=$2
                    shift 2
                else
                    ampere_gpus=1
                    shift
                fi
                ;;
            --name|-n)
                job_name="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Usage: sa [--ampere|-a [num_gpus]] [--name|-n job_name]" >&2
                return 1
                ;;
        esac
    done

    # Set default job name if not provided
    if [ -z "$job_name" ]; then
        if [ "$ampere_gpus" -gt 0 ]; then
            job_name="$default_ampere_name"
        else
            job_name="$default_regular_name"
        fi
    fi

    # --- Check for existing job ---
    local existing_id
    existing_id=$(_get_job_id_by_name "$job_name")

    if [ ! -z "$existing_id" ]; then
        echo "Error: Job '$job_name' is already running (ID: $existing_id)." >&2
        echo "Connect to it with: sr $job_name" >&2
        return 1
    fi

    # --- Step 1: Allocate the job WITHOUT a shell to capture its ID ---
    local salloc_cmd="salloc --no-shell -J \"$job_name\" -n 1 -c 4 --mem 24G -t 22:00:00"
    local salloc_output
    
    echo "Attempting to allocate job: $job_name..."
    if [ "$ampere_gpus" -gt 0 ]; then
        salloc_output=$($salloc_cmd -p gpu_quad,gpu,gpu_requeue --qos=gpuquad_qos --gres=gpu:$ampere_gpus --nodes=1 --nodelist="compute-gc-17-[249,252-254,239-240],compute-g-17-[162-163,166-171,200-205]" 2>&1)
    else
        salloc_output=$($salloc_cmd -p gpu_quad,gpu,gpu_requeue --qos=gpuquad_qos --gres=gpu:1 -x "compute-g-17-164" 2>&1)
    fi

    # --- Step 2: Parse the Job ID from the output ---
    local job_id
    job_id=$(echo "$salloc_output" | grep "Granted job allocation" | awk '{print $5}')

    if [ -z "$job_id" ]; then
        echo "Error: Failed to allocate job." >&2
        echo "$salloc_output" >&2
        return 1
    fi

    echo "Successfully allocated job '$job_name' (ID: $job_id)"

    # --- Step 3: Start the main interactive shell (this blocks) ---
    srun --jobid="$job_id" --pty /bin/bash

    # --- Step 4: Clean up message when the shell exits ---
    echo "Allocation $job_id ($job_name) finished."
}

# --- (sr) srun: Connects to an existing allocation ---
function sr() {
    local job_name="$1"
    
    # Set default job name if not provided
    if [ -z "$job_name" ]; then
        job_name="main-interactive"
    fi

    # --- Find the job ID ---
    local job_id
    job_id=$(_get_job_id_by_name "$job_name")

    if [ -z "$job_id" ]; then
        echo "Error: No active job named '$job_name' found." >&2
        echo "Start one with: sa -n $job_name" >&2
        return 1
    fi

    echo "Connecting to job '$job_name' (ID: $job_id)..."
    
    srun --jobid="$job_id" --overlap --pty /bin/bash
}

# --- Tab completion for sr (srun) function ---
function _sr_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    
    # Get all job names from squeue
    local squeue_output
    squeue_output=$(squeue -u joj144 --Format="JobID,Name:100" --noheader 2>/dev/null)
    
    if [ -z "$squeue_output" ]; then
        COMPREPLY=()
        return 0
    fi
    
    # Extract job names from squeue output
    local job_names=()
    while IFS= read -r line; do
        local line_job_name
        line_job_name=$(echo "$line" | awk -F'"' '{print $2}')
        if [ ! -z "$line_job_name" ]; then
            job_names+=("$line_job_name")
        fi
    done <<< "$squeue_output"
    
    # Filter job names that start with what the user has typed
    local matches=()
    for name in "${job_names[@]}"; do
        if [[ "$name" == "$cur"* ]]; then
            matches+=("$name")
        fi
    done
    
    COMPREPLY=("${matches[@]}")
}

# Register the completion function for 'sr'
complete -F _sr_completion sr


# zoxide - smarter cd command
eval "$(zoxide init bash)"

#THIS MUST BE AT THE END OF THE FILE FOR SDKMAN TO WORK!!!
export SDKMAN_DIR="$HOME/.sdkman"
[[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"

[ -f ~/.fzf.bash ] && source ~/.fzf.bash
