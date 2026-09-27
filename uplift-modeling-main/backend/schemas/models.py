from pydantic import BaseModel, Field

class BudgetRequest(BaseModel):
    budget: float = Field(ge=0)
    cost_per_contact: float = Field(ge=0)
    revenue_per_incremental_conversion: float = Field(ge=0)

class PolicyRequest(BaseModel):
    revenue_per_incremental_conversion: float = Field(default=50, ge=0)
    cost_per_contact: float = Field(default=.05, ge=0)

class CustomerListQuery(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    segment: str | None = None


class GuardrailRequest(BaseModel):
    budget: float = Field(ge=0)
    cost_per_contact: float = Field(ge=0)
    revenue_per_incremental_conversion: float = Field(ge=0)
    min_uplift: float = Field(default=0, ge=-1, le=1)
    max_contacts: int | None = Field(default=None, ge=1, le=64000)
    exclude_segments: list[str] = Field(default_factory=list)

