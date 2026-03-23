# Development Process

## Git Branching Strategy

We use a simplified **Git Flow** adapted for academic group work:

```
main
 └── develop
      ├── feature/core-grid
      ├── feature/explicit-solver
      ├── feature/crank-nicolson
      ├── feature/gray-scott-validation
      ├── feature/giraffe-pattern
      └── feature/leopard-pattern
```

### Branch Rules

| Branch | Purpose | Who merges? |
|--------|---------|-------------|
| `main` | Stable, tagged releases | After milestone review |
| `develop` | Working integration | Any member via MR |
| `feature/*` | Single feature / task | Author opens MR |

### Commit Message Convention (Conventional Commits)

```
<type>: <short description>

Types:
  feat     – new feature
  fix      – bug fix
  test     – adding/fixing tests
  docs     – documentation only
  refactor – code restructuring (no behavior change)
  perf     – performance improvement
  chore    – tooling, CI, dependencies
```

### Workflow Steps

1. Pull latest `develop`:
   ```bash
   git checkout develop && git pull
   ```
2. Create feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Develop, commit often:
   ```bash
   git add .
   git commit -m "feat: implement explicit euler solver"
   ```
4. Push and open Merge Request:
   ```bash
   git push origin feature/my-feature
   ```
5. Request code review from one team member.
6. Merge into `develop` after approval.

### Recommended Workflow Cycle (per sprint)

```
Plan Issues → Assign → Branch → Implement → Test → MR → Review → Merge
```

## Testing Policy

- All new code must include unit tests.
- Tests must pass before merging into `develop`.
- Run tests locally: `pytest tests/ -v`

## Code Style

- Follow PEP 8.
- Use type hints for all function signatures.
- Use NumPy docstring format for all public functions and classes.
