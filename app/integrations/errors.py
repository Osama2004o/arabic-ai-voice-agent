"""Shared exception type for external integrations."""


class IntegrationError(Exception):
    """Raised when an external integration fails.

    The message is safe to return to clients: it never contains credentials or tokens.
    `status_code` is the HTTP status the API layer should respond with.
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
