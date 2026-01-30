# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions
alias ..='cd ..'
alias nv='nvim'
alias ta='tmux attach'

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

# 5. Vim-style history navigation (Ctrl+K = up, Ctrl+J = down)
if [[ $- == *i* ]]; then
    bind '"\C-k": previous-history'
    bind '"\C-j": next-history'
fi

# Locale - use en_US.UTF-8 if available, otherwise fall back to C.UTF-8
if locale -a 2>/dev/null | grep -q "en_US.utf8"; then
    export LANG=en_US.UTF-8
elif locale -a 2>/dev/null | grep -q "C.utf8\|C.UTF-8"; then
    export LANG=C.UTF-8
fi
export LC_ALL=$LANG

export VSCODE_SERVER_DIR=$HOME/.vscode-server

# PATH
export PATH=$HOME/bin:$PATH
export PATH=$PATH:$HOME/local/go/bin
export PATH=$PATH:$HOME/go/bin
export PATH=$HOME/.local/nvim/bin:$PATH
export PATH=$HOME/gopath/bin:$PATH

export UV_INDEX_STRATEGY=first-index

function g() {
	gemini -p "$*"
}

function dt() {
	# Check login status first; if expired/invalid, login
	if ! devtunnel user show &>/dev/null; then
		echo "Login required. Logging in..."
		devtunnel user login
	fi
	devtunnel host -p "$1" --allow-anonymous
}

function claude() {
	local args=()
	local skip_permissions=true

	for arg in "$@"; do
		if [[ "$arg" == "--dont-skip-permissions" ]]; then
			skip_permissions=false
		else
			args+=("$arg")
		fi
	done

	if $skip_permissions; then
		command claude --dangerously-skip-permissions "${args[@]}"
	else
		command claude "${args[@]}"
	fi
}

function cr() {
    # Replace both '/' and '_' with '-'
    local project_path="$HOME/.claude/projects/$(pwd | tr '/_' '-')"
    
    if [[ ! -d "$project_path" ]]; then
        echo "Error: Claude project directory not found:"
        echo "$project_path"
        return 1
    fi

    ls -lst "$project_path" | \
    tail -n +2 | \
    fzf --header="Select conversation to resume" | \
    awk '{print $NF}' | \
    sed -E 's/\.jsonl?$//' | \
    xargs -r -I {} claude --dangerously-skip-permissions -r {}
}

function wp() {
    # Get the absolute path of the argument (or current dir if empty)
    local target_path=$(readlink -f "${1:-.}/")

    # 1. Swap the O2 prefix for the WSL/sshfs prefix
    # 2. Convert all forward slashes to backslashes
    echo "$target_path" | \
    sed 's|^/n/groups/datta|\\\\wsl.localhost\\wsl-d\\home\\jrdja\\o2_mnt|' | \
    sed 's|/|\\|g'
}

function nav() {
    while true; do
        local sel=$(ls -a | fzf --preview 'if [ -d {} ]; then ls -la {}; else head -100 {}; fi')
        [[ -z "$sel" ]] && return  # cancelled with Esc
        if [[ -d "$sel" ]]; then
            cd "$sel"
        elif [[ -f "$sel" ]]; then
            nvim "$sel"
            return
        fi
    done
}

# quickdir - terminal directory browser
b() {
    export QUICKDIR_LASTDIR=$(mktemp)
    quickdir "$@"
    if [ -f "$QUICKDIR_LASTDIR" ]; then
        cd "$(cat "$QUICKDIR_LASTDIR")"
        rm -f "$QUICKDIR_LASTDIR"
    fi
    unset QUICKDIR_LASTDIR
}

# Cursor/VSCode: disable conda prompt modifications for clean agent prompts
if [[ ${TERM_PROGRAM:-} == "vscode" || ${TERM_PROGRAM:-} == "cursor" ]]; then
  export CONDA_CHANGEPS1=no
  export MAMBA_NO_CHANGEPS1=1
  export CONDA_PROMPT_MODIFIER=""
  unset PROMPT_COMMAND
  trap - DEBUG
  export PS1='[\u@\h \W]\$ '
fi

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba shell init' !!
export MAMBA_EXE="$HOME/miniforge3/bin/mamba";
export MAMBA_ROOT_PREFIX="$HOME/miniforge3";
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
__conda_setup="$("$HOME/miniforge3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# zoxide - smarter cd command
command -v zoxide &> /dev/null && eval "$(zoxide init bash)"

#THIS MUST BE AT THE END OF THE FILE FOR SDKMAN TO WORK!!!
export SDKMAN_DIR="$HOME/.sdkman"
[[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"

[ -f ~/.fzf.bash ] && source ~/.fzf.bash

# Source machine-specific config if it exists
[[ -f ~/.bashrc.local ]] && source ~/.bashrc.local
. "$HOME/.cargo/env"
