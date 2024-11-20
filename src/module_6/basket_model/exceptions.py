class UserNotFoundException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return f"UserNotFoundException"


class PredictionException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return f"PredictionException"
