from pydantic import BaseModel, constr, confloat

class PredictPricePayload(BaseModel):
    carat: confloat(gt=0)
    cut: constr(min_length=1)
    color: constr(min_length=1)
    clarity: constr(min_length=1)
    depth: confloat(gt=0)
    table: confloat(gt=0)
    x: confloat(gt=0)
    y: confloat(gt=0)
    z: confloat(gt=0)

class SimilarDiamondsPayload(BaseModel):
    carat: confloat(gt=0)
    cut: constr(min_length=1)
    color: constr(min_length=1)
    clarity: constr(min_length=1)