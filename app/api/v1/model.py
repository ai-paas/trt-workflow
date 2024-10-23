from fastapi import APIRouter
from sqlalchemy.orm import Session

from app.core.db.connect import SessionDepends
from app.schemas.requests import model
from app.services.model_service import ModelService

router = APIRouter()

"""
모델 이름을 입력받고 해당 모델을 경량화/최적화 후 mlflow에 저장하는 라우터
"""


@router.post("")
def lite_model(
    *,
    db: Session = SessionDepends,
    lite_model_form: model.LiteModelForm,
):
    """
    * params
        - lite_type
            2. trt
    """
    result = ModelService.lite_model_trt(db=db, model_create_form=lite_model_form)
    return result
