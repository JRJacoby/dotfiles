-- ~/.config/nvim/init.lua

-- This is the standard "bootstrap" code for lazy.nvim
-- It will automatically install lazy.nvim if it's not found
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not (vim.uv or vim.loop).fs_stat(lazypath) then
  vim.fn.system({
    "git",
    "clone",
    "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable", -- latest stable release
    lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)

-- Make 'y' (yank) and 'p' (paste) use the system clipboard
vim.opt.clipboard = 'unnamedplus'

-- Enable relative line numbers (with absolute number on current line)
vim.opt.number = true
vim.opt.relativenumber = true

-- This is where you list your plugins
require("lazy").setup({

  -- YOUR FIRST PLUGIN: nvim-treesitter
  -- This provides fast and accurate syntax highlighting
  {
    "nvim-treesitter/nvim-treesitter",
    build = ":TSUpdate", -- Installs parsers on update
    config = function()
      -- This config function runs after the plugin loads
      require("nvim-treesitter.configs").setup({

        -- A list of parser names, or "all"
        -- We're installing markdown, lua (for this config), and vimdoc (for help files)
        ensure_installed = { "markdown", "markdown_inline", "lua", "vimdoc", "python" },

        -- Install parsers synchronously (blocks startup)
        sync_install = false,

        -- Automatically install missing parsers when entering a buffer
        auto_install = true,

        -- Enable syntax highlighting
        highlight = {
          enable = true,
        },
      })
    end,
  },

  -- == ADDED PLUGINS START HERE ==

  -- PLUGIN 2: telescope.nvim (Fuzzy Finder)
  {
    'nvim-telescope/telescope.nvim',
    tag = '0.1.6', -- Pinning to a recent stable tag
    dependencies = { 'nvim-lua/plenary.nvim' },
    config = function()
      require('telescope').setup({
        defaults = {
          file_ignore_patterns = { ".git/" },
        },
        pickers = {
          find_files = {
            hidden = true,
          },
        },
      })
      require('telescope').load_extension('project')
      require('telescope').load_extension('session-lens')
    end
  },

  -- PLUGIN 3: telescope-project (Project Manager for Telescope)
  {
    "nvim-telescope/telescope-project.nvim",
    dependencies = { "nvim-telescope/telescope.nvim" }
  },

  -- PLUGIN 4: neo-tree.nvim (File Explorer Sidebar)
  {
    "nvim-neo-tree/neo-tree.nvim",
    branch = "v3.x",
    dependencies = {
      "nvim-lua/plenary.nvim",
      "nvim-tree/nvim-web-devicons", -- Optional: for file icons
      "MunifTanjim/nui.nvim",
    },
    config = function()
      require("neo-tree").setup({
        filesystem = {
          filtered_items = {
            visible = true,
            hide_dotfiles = false,
            hide_gitignored = false,
          },
        },
      })
    end
  },

  -- PLUGIN 5: oil.nvim (File Explorer as Buffer)
  {
    "stevearc/oil.nvim",
    dependencies = { "nvim-tree/nvim-web-devicons" },
    config = function()
      require("oil").setup({
        view_options = {
          show_hidden = true,
        },
      })
    end
  },

  -- PLUGIN 6: nvim-lspconfig (provides server configs for vim.lsp)
  {
    "neovim/nvim-lspconfig",
    config = function()
      -- Set up keymaps when an LSP attaches to a buffer
      vim.api.nvim_create_autocmd("LspAttach", {
        callback = function(args)
          local opts = { buffer = args.buf }
          vim.keymap.set("n", "gd", vim.lsp.buf.definition, opts)
          vim.keymap.set("n", "gD", vim.lsp.buf.declaration, opts)
          vim.keymap.set("n", "gr", vim.lsp.buf.references, opts)
          vim.keymap.set("n", "gi", vim.lsp.buf.implementation, opts)
          vim.keymap.set("n", "K", vim.lsp.buf.hover, opts)
          vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename, opts)
          vim.keymap.set("n", "<leader>ca", vim.lsp.buf.code_action, opts)
        end,
      })

      -- Python (pyright) - using new vim.lsp API
      vim.lsp.config('pyright', {})
      vim.lsp.enable('pyright')
    end
  },

  -- PLUGIN 8: onedark.nvim (Colorscheme)
  {
    "navarasu/onedark.nvim",
    priority = 1000, -- Load before other plugins
    config = function()
      require("onedark").setup({
        style = "dark", -- Options: dark, darker, cool, deep, warm, warmer
      })
      require("onedark").load()
    end
  },

  -- PLUGIN 9: vim-zoom (Toggle zoom on a window)
  {
    "dhruvasagar/vim-zoom",
    keys = { { "<C-w>m", "<Plug>(zoom-toggle)", desc = "Toggle Zoom" } },
  },

  -- PLUGIN 10: auto-session (Per-directory session management)
  {
    "rmagatti/auto-session",
    lazy = false,
    config = function()
      require("auto-session").setup({
        suppressed_dirs = { "~/", "~/Downloads", "/" },
      })

      -- Periodic session save every 15 minutes (for HPC job timeouts)
      vim.fn.timer_start(900000, function()
        require("auto-session").auto_save_session()
      end, { ["repeat"] = -1 })
    end
  },

  -- == ADDED PLUGINS END HERE ==

  -- Add more plugins here in the future
  -- { "another-github-user/another-plugin" },

})

-- == KEYMAPS START HERE ==

-- Set <space> as the leader key
-- NOTE: Must be set BEFORE setting any keymaps
vim.g.mapleader = " "
vim.g.maplocalleader = " "

-- A helper function to make keymaps easier to read
local keymap = vim.keymap.set

-- Keymaps for Neo-Tree (File Explorer)
-- "n" means "normal mode"
-- "<leader>e" means "Space + e"
keymap("n", "<leader>e", ":Neotree toggle<CR>", { desc = "Toggle File Explorer" })

-- Keymaps for Telescope (Fuzzy Finder)
keymap("n", "<leader>ff", "<cmd>Telescope find_files<cr>", { desc = "Find Files (in current project)" })
keymap("n", "<leader>fg", "<cmd>Telescope live_grep<cr>", { desc = "Grep Text (in current project)" })
keymap("n", "<leader>fb", "<cmd>Telescope buffers<cr>", { desc = "Find Open Buffers (Tabs)" })

-- Keymap for Telescope Project (Multi-Root Workspace)
keymap("n", "<leader>fp", "<cmd>Telescope project<cr>", { desc = "Find Projects" })

-- Keymap for Telescope Sessions (auto-session)
keymap("n", "<leader>fs", "<cmd>Telescope session-lens<cr>", { desc = "Find Sessions" })

-- More Telescope keymaps
keymap("n", "<leader>fh", "<cmd>Telescope help_tags<cr>", { desc = "Find Help" })
keymap("n", "<leader>fo", "<cmd>Telescope oldfiles<cr>", { desc = "Find Recent Files" })
keymap("n", "<leader>fw", "<cmd>Telescope grep_string<cr>", { desc = "Find Word Under Cursor" })
keymap("n", "<leader>fd", "<cmd>Telescope diagnostics<cr>", { desc = "Find Diagnostics" })
keymap("n", "<leader>fr", "<cmd>Telescope resume<cr>", { desc = "Resume Last Search" })
keymap("n", "<leader>fk", "<cmd>Telescope keymaps<cr>", { desc = "Find Keymaps" })
keymap("n", "<leader>/", "<cmd>Telescope current_buffer_fuzzy_find<cr>", { desc = "Fuzzy Find in Buffer" })

-- Keymap for Oil (File Explorer as Buffer)
keymap("n", "-", "<cmd>Oil<cr>", { desc = "Open Parent Directory" })

-- Diagnostic keymaps
keymap("n", "gl", vim.diagnostic.open_float, { desc = "Show Diagnostic Details" })
keymap("n", "[d", vim.diagnostic.goto_prev, { desc = "Previous Diagnostic" })
keymap("n", "]d", vim.diagnostic.goto_next, { desc = "Next Diagnostic" })

-- Window navigation (works in normal and terminal mode)
keymap("n", "<C-h>", "<C-w>h", { desc = "Move to Left Window" })
keymap("n", "<C-j>", "<C-w>j", { desc = "Move to Lower Window" })
keymap("n", "<C-k>", "<C-w>k", { desc = "Move to Upper Window" })
keymap("n", "<C-l>", "<C-w>l", { desc = "Move to Right Window" })

-- Terminal keymaps
keymap("n", "<leader>th", "<cmd>split | terminal<cr>", { desc = "Terminal (horizontal split)" })
keymap("n", "<leader>tv", "<cmd>vsplit | terminal<cr>", { desc = "Terminal (vertical split)" })
keymap("n", "<leader>tt", "<cmd>terminal<cr>", { desc = "Terminal (current window)" })

-- Auto-enter terminal mode when switching to a terminal buffer
vim.api.nvim_create_autocmd({"BufEnter", "WinEnter"}, {
  callback = function()
    if vim.bo.buftype == "terminal" then
      vim.cmd("startinsert")
    end
  end
})
keymap("t", "<Esc><Esc>", [[<C-\><C-n>]], { desc = "Exit terminal mode" })
keymap("t", "<C-\\>", [[<C-\><C-n>]], { desc = "Exit terminal mode" })
keymap("t", "<C-w>h", [[<C-\><C-n><C-w>h]], { desc = "Move to Left Window" })
keymap("t", "<C-w>j", [[<C-\><C-n><C-w>j]], { desc = "Move to Lower Window" })
keymap("t", "<C-w>k", [[<C-\><C-n><C-w>k]], { desc = "Move to Upper Window" })
keymap("t", "<C-w>l", [[<C-\><C-n><C-w>l]], { desc = "Move to Right Window" })

-- Broadcast command to all terminal windows in current tab
vim.api.nvim_create_user_command("Bc", function(opts)
  local cmd = opts.args
  local current_win = vim.api.nvim_get_current_win()
  for _, win in ipairs(vim.api.nvim_tabpage_list_wins(0)) do
    local buf = vim.api.nvim_win_get_buf(win)
    if vim.bo[buf].buftype == "terminal" then
      local job_id = vim.b[buf].terminal_job_id
      if job_id then
        vim.fn.chansend(job_id, cmd .. "\r")
      end
    end
  end
  vim.api.nvim_set_current_win(current_win)
end, { nargs = 1, desc = "Broadcast command to all terminals in tab" })

-- == KEYMAPS END HERE ==
