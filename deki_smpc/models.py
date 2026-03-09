"""Pydantic request/registration models used by client-server APIs."""

from pydantic import BaseModel


class KeyClientRegistration(BaseModel):
    """Client registration payload for the key aggregation service."""

    ip_address: str
    client_name: str
    preshared_secret: str


class CheckForTaskRequest(BaseModel):
    """Payload for polling task assignments for a specific client."""

    client_name: str
