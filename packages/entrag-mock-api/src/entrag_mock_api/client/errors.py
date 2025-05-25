class MockAPIError(Exception):
    """
    Base exception for Mock API client errors.
    """

    pass


class MockAPIConnectionError(MockAPIError):
    """
    Raised when unable to connect to the API.
    """

    pass


class MockAPINotFoundError(MockAPIError):
    """
    Raised when requested resource is not found.
    """

    pass


class MockAPIValidationError(MockAPIError):
    """
    Raised when API response validation fails.
    """

    pass
