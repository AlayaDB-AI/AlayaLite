# Contributing to AlayaLite

Thank you for your interest in contributing to AlayaLite! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and inclusive community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/AlayaLite.git
   cd AlayaLite
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/AlayaDB-AI/AlayaLite.git
   ```

## Development Setup

### Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake >= 3.15
- Python >= 3.8
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Conan 2.x (C++ package manager)

### Build from Source

```bash
# Install Python dev dependencies
make dev-install

# Build release version
make build

# Run tests
make test
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them before making changes:

```bash
uvx pre-commit install
uvx pre-commit install --hook-type commit-msg
```

## Making Changes

1. **Create a new branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** and ensure:
   - Code follows the project style guidelines
   - All tests pass (`make test`)
   - Linting passes (`make lint`)

3. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

The `(<scope>)` segment is **required**. The CI and local `commit-msg` hook
will reject any commit or PR title that omits it. The *content* of the scope is
not whitelisted — see [Scopes](#scopes) below for the recommended names and
the conventions for cross-cutting changes.

### Allowed Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring |
| `perf` | Performance improvements |
| `test` | Adding or updating tests |
| `build` | Build system changes |
| `ci` | CI/CD configuration changes |
| `chore` | Other changes (dependencies, etc.) |
| `revert` | Revert a previous commit |

The allowed types are enforced by `conventional-pre-commit` (see
`.pre-commit-config.yaml`). A commit with any other type will be rejected.

### Scopes

Scopes name the **module** the change touches. The CI only verifies that *some*
scope is present — it does not constrain the scope value. The lists below are
**recommended** names; please reuse them rather than inventing new spellings,
so `git log --grep` and changelog tooling stay useful.

When in doubt, use the directory name as the scope. That keeps the mapping
mechanical and unambiguous.

#### Recommended scopes

C++ core (`include/`):

| Scope | Path |
|-------|------|
| `index` | `include/index/` (general graph indices, e.g. HNSW, Vamana) |
| `disk` | `include/index/disk/` |
| `laser` | `include/index/graph/laser/` |
| `executor` | `include/executor/` |
| `recovery` | `include/recovery/` |
| `simd` | `include/simd/` |
| `space` | `include/space/` |
| `storage` | `include/storage/` |
| `utils` | `include/utils/` |

Python and apps:

| Scope | Path |
|-------|------|
| `python` | `python/src/alayalite/` (Python SDK, bindings) |
| `bench` | `python/src/alayalite/bench/`, `python/tests/bench/` |
| `app` | `app/` |
| `examples` | `examples/` |

Build / CI / docs (use as a narrowing scope):

| Scope | Use for |
|-------|---------|
| `cmake` | CMake configuration changes (`build(cmake): ...`) |
| `conan` | Conan dependency or recipe changes |
| `cibuildwheel` | Wheel-build workflow tweaks |
| `readme` | Top-level README updates |

Cross-cutting / repo-wide (use one of these when no module dominates):

| Scope | Use for |
|-------|---------|
| `deps` | Dependency bumps (`chore(deps):`, `build(deps):`) — matches Dependabot's default |
| `repo` | Repository-wide chores: top-level configs, `.gitignore`, license, governance |
| `release` | Version bumps, changelog, release tagging |

#### Multiple scopes

When a single logical change genuinely spans 2–3 modules, list them
comma-separated in **alphabetical order**, no spaces:

```
feat(disk,laser,python): expose batch_search through Python bindings
```

If the list would grow beyond 3 scopes, the change is no longer localized —
pick the most representative single scope, or use a cross-cutting scope like
`repo`.

### Examples

Single scope (most common):

```bash
feat(laser): warmup path fix and PCA write optimization
fix(index): correct neighbor pruning in Vamana build
perf(simd): vectorize L2 distance for AVX-512
test(bench): consolidate benchmark smoke tests
docs(laser): document RaBitQ tuning knobs
```

Multiple scopes (alphabetical, comma-separated):

```bash
feat(disk,laser,python): expose batch_search through Python bindings
refactor(space,utils): extract shared metric helpers
```

Cross-cutting / repo-wide:

```bash
chore(deps): bump pybind11 to 2.13.6
build(cmake): enable LTO for release builds
ci(cibuildwheel): skip Python 3.8 on Windows
docs(readme): update installation instructions
chore(repo): add CODE_OF_CONDUCT and templates
```

## Pull Request Process

1. **Create an issue first** to discuss the feature or bug fix
2. **Ensure all checks pass**:
   - CI builds successfully
   - All tests pass
   - Code coverage is maintained
3. **Update documentation** if needed
4. **Request review** from maintainers
5. **Address feedback** and update your PR as needed

### PR Title Format

Follow the same convention as commit messages, including the required scope:

```
feat(space): add support for cosine similarity metric
fix(disk): correct page eviction under concurrent search
docs(repo): add community documentation and templates
```

The PR title becomes the squash-merge commit on the base branch, so the scope
rules above apply to it directly.

## Code Style

### C++ Style

- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with modifications
- Use `clang-format` (automatically applied via pre-commit)
- Maximum line length: 100 characters
- Use C++20 features where appropriate

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use `ruff` for formatting and linting
- Use type hints for function signatures

### File Organization

```
include/           # C++ header-only library
  ├── index/       # Index implementations
  ├── space/       # Vector space abstractions
  ├── simd/        # SIMD optimizations
  └── utils/       # Utility functions

python/
  ├── src/         # Python SDK source
  └── tests/       # Python tests

tests/             # C++ tests
```

## Questions?

If you have any questions, feel free to:
- Open an issue on GitHub
- Contact us at dev@alayadb.ai

Thank you for contributing to AlayaLite!
