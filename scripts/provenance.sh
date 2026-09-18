#!/bin/bash

# Check source/build provenance for normal jobs. The caller must check that the
# executable and its build stamp exist before invoking this function.
murb_check_provenance() {
    local root="$1"
    local bin="$2"
    local strict="${MURB_STRICT_PROVENANCE:-0}"
    local worktree_status
    local tracked_dirty=0
    local provenance_issue=0
    local executable_revision
    local executable_dirty

    [[ "$strict" == "0" || "$strict" == "1" ]] || {
        echo "MURB_STRICT_PROVENANCE must be 0 or 1: $strict" >&2
        return 1
    }

    MURB_GIT_REVISION="$(git -C "$root" rev-parse HEAD)" || return 1
    MURB_EXECUTABLE_VERSION="$("$bin" --version)" || return 1

    if [[ "$MURB_EXECUTABLE_VERSION" =~ (^|[[:space:]])revision=([^[:space:]]+) ]]; then
        executable_revision="${BASH_REMATCH[2]}"
    else
        echo "Incompatible executable (missing revision in --version): $MURB_EXECUTABLE_VERSION" >&2
        return 1
    fi
    if [[ "$MURB_EXECUTABLE_VERSION" =~ (^|[[:space:]])dirty=([01])([[:space:]]|$) ]]; then
        executable_dirty="${BASH_REMATCH[2]}"
    else
        echo "Incompatible executable (missing dirty status in --version): $MURB_EXECUTABLE_VERSION" >&2
        return 1
    fi

    worktree_status="$(git -C "$root" status --porcelain --untracked-files=normal)" || return 1
    if git -C "$root" diff --quiet HEAD --; then
        tracked_dirty=0
    else
        local diff_status=$?
        [[ "$diff_status" -eq 1 ]] || return "$diff_status"
        tracked_dirty=1
    fi

    echo "Git HEAD: $MURB_GIT_REVISION"
    echo "Git worktree dirty: $([[ -n "$worktree_status" ]] && echo 1 || echo 0)"
    echo "Executable version: $MURB_EXECUTABLE_VERSION"
    echo "Executable build dirty: $executable_dirty"

    if [[ -n "$worktree_status" ]]; then
        echo "WARNING: Git worktree is dirty; reproducibility is not guaranteed." >&2
        provenance_issue=1
    fi
    if [[ "$tracked_dirty" -eq 1 ]]; then
        echo "WARNING: tracked files differ from HEAD; the executable may not correspond exactly to the current source tree." >&2
        provenance_issue=1
    fi
    if [[ "$executable_dirty" -eq 1 ]]; then
        echo "WARNING: executable was built from a dirty source tree." >&2
        provenance_issue=1
    fi
    if [[ "$executable_revision" != "$MURB_GIT_REVISION" ]]; then
        echo "WARNING: executable revision $executable_revision differs from current HEAD $MURB_GIT_REVISION; the executable may not correspond exactly to the current source tree." >&2
        provenance_issue=1
    fi

    if [[ "$strict" == "1" && "$provenance_issue" -eq 1 ]]; then
        echo "Strict provenance requested with MURB_STRICT_PROVENANCE=1; refusing to continue." >&2
        return 1
    fi
}
