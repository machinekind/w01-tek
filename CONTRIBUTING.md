# Contributing to Wojtek

Start with the [documentation home](docs/README.md). Read [AGENTS.md](AGENTS.md)
before changing training, models, experiments or deployment. Experiments stay
outside the production dependency graph. Do not commit credentials, private machine
identities, generated runs, weights or recordings.

## Validate a change

From the repository root, create the docs environment once:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install --require-hashes -r scripts/requirements-docs.txt
```

Then run:

```bash
.venv-docs/bin/python -m unittest discover -s scripts -p 'test_*.py'
.venv-docs/bin/python scripts/check_docs.py
git diff --check
```

The checker uses CommonMark parsing with tables and strikethrough enabled. It
checks local inline/reference links, images, heading anchors and explicit HTML
`<a id/name>` anchors, and compares runtime defaults against configuration.
Code examples are excluded. Raw HTML links, external URLs and other GitHub
extensions are outside its scope; commands needing ROS, a GPU or hardware need
their own validation.

For training Python changes, use the installed environment:

```bash
./training/run.sh train --cfg job --resolve
./training/run.sh test
```

For model-generation changes, also run `./training/run.sh build` and
`./training/run.sh check`, then inspect generated XML before committing it.
Changes under an experiment need `./experiments/<experiment>/run.sh test` as
described in that experiment's README.

Environment/model/randomization changes additionally need the integration suite;
see the complete [validation matrix](AGENTS.md#validation-by-change-type).
Do not run CPU training as a smoke test. For ROS changes,
run the affected package's tests in the ROS development container. Pure modules
can also be tested from a Python environment with their declared dependencies;
that does not validate ROS launch composition or hardware behavior.

## Write for a reader with a task

Use English for new entry points and operating guides. Preserve useful Polish
technical notes and link them with their language stated. Keep one current
reference for shared facts; link it instead of copying rates or control mappings
into several guides.

A tutorial needs prerequisites, working directory, commands, expected output,
completion criteria and recovery/shutdown. State whether commands run on the
host, inside the container or on the robot. Use placeholders for personal
infrastructure and identify policy access requirements before the first launch.

Distinguish implemented, measured, unsupported and unvalidated behavior. A dated
experiment is evidence for its recorded setup. It does not become a current
default or a result for another robot revision. Preserve historical results with
their date; replace superseded instructions in current guides.

## Review and ownership

The author of a code or hardware change updates the affected contract and guide
in the same PR. The reviewer responsible for that subsystem reviews its docs:
control/ROS, learning, mechanics, electronics or operator tooling. This is role
ownership; named maintainers can be assigned when the team chooses them.

| Change | Documentation to check |
|---|---|
| Model, leg revision, joint map, policy observations | Compatibility, variants, training, twin record |
| Topics, services, QoS, timeouts | Architecture and integration example |
| Launch defaults or service configuration | Compatibility, simulation, operations, troubleshooting |
| Controls or recovery behavior | Operations and Deck guide |
| Harness, sensors, drive firmware | Hardware, commissioning, evidence register |

Keep code fixes separate from broad prose rewrites when they can be reviewed
independently. Each PR records the checks actually run and the remaining limits.
The [validation record](docs/validation.md) includes walkthroughs still requiring
a fresh reader or physical unit. Do not mark those complete from a static review.
