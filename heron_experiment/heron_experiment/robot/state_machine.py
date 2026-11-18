#!/usr/bin/env python3
"""Generic event-driven state machine: (state, event) -> state, with an
optional async on-enter handler per state and an optional callback fired on
every state change. Knows nothing about ROS, a robot, or any particular
domain — the transition table and handlers are supplied by the caller, so the
same class backs any node's explicit state machine (e.g. grasp.py's pick
sequence).
"""


class StateMachine:
    """`transitions` is a dict of (state, event) -> next_state, evaluated in
    `fire()`. `handlers` optionally maps a state to an async callable invoked
    right after entering it (e.g. to kick off the work that state performs).
    `on_state_changed` is optionally invoked with the new state on every
    transition, including once for the initial state at construction time —
    hook a state-topic publisher there instead of writing to it at every
    call site.
    """

    def __init__(self, transitions, initial_state, handlers=None,
                 on_state_changed=None, logger=None):
        self._transitions = transitions
        self._handlers = handlers or {}
        self._on_state_changed = on_state_changed
        self._logger = logger
        self._state = initial_state
        if self._on_state_changed is not None:
            self._on_state_changed(self._state)

    @property
    def state(self) -> str:
        return self._state

    def reset(self, state: str) -> None:
        """Force the current state without going through a transition or
        running a handler — for abort/exception paths that must snap back to
        a known state (e.g. idle) outside the normal event flow."""
        self._state = state
        if self._on_state_changed is not None:
            self._on_state_changed(self._state)

    async def fire(self, event: str) -> None:
        """Look up (state, event) and transition, running the next state's
        handler if one is registered. Unknown event for the current state:
        log and no-op (never raises)."""
        key = (self._state, event)
        next_state = self._transitions.get(key)
        if next_state is None:
            if self._logger is not None:
                self._logger.warn(
                    f"No transition for event '{event}' in state '{self._state}' — ignored")
            return
        if self._logger is not None:
            self._logger.info(f"{self._state} --{event}--> {next_state}")
        self._state = next_state
        if self._on_state_changed is not None:
            self._on_state_changed(next_state)
        handler = self._handlers.get(next_state)
        if handler is not None:
            await handler()
