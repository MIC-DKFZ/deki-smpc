from pydantic import BaseModel


class KeyClientRegistration(BaseModel):
    ip_address: str
    client_name: str
    preshared_secret: str
