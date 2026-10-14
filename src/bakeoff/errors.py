"""Error budget for a measurement pass."""


class AbortRun(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ErrorBudget:
    def __init__(self, pass_name: str, threshold: float = 0.2, min_requests: int = 10):
        self.pass_name = pass_name
        self.threshold = threshold
        self.min_requests = min_requests
        self.n = 0
        self.errors = 0

    def record(self, ok: bool) -> None:
        self.n += 1
        if not ok:
            self.errors += 1
        if self.n >= self.min_requests and self.errors / self.n > self.threshold:
            raise AbortRun(f"{self.pass_name}: {self.errors}/{self.n} requests failed")
