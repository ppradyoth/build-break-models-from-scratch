.PHONY: check solve test lint gaps
LEVEL ?= 01

## Run the level against YOUR starter code (fails until you fill the gaps)
check:
	BBM_IMPL=starter pytest levels/level-$(LEVEL)-*/tests -v

## Run the level against the reference solution (should be all green)
solve:
	BBM_IMPL=solution pytest levels/level-$(LEVEL)-*/tests -v

## Full reference suite (what CI runs)
test:
	BBM_IMPL=solution pytest -q

## Pedagogy invariant: every starter still has its gaps
gaps:
	@! grep -rL "NotImplementedError" levels/*/starter/*.py || (echo "gap markers present" && true)

lint:
	ruff check ctf levels tests
