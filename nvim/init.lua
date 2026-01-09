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
        ensure_installed = { "markdown", "markdown_inline", "lua", "vimdoc" },

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
      -- This config function runs after the plugin loads
      -- We also load the 'project' extension here
      require('telescope').load_extension('project')
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
      -- You can add neo-tree setup options here in the future
      -- For now, it will just load with default settings
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

-- == KEYMAPS END HERE ==
