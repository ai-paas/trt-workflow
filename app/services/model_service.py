from uuid import uuid4

from kfp import dsl
from sqlalchemy.orm import Session

from app.config.model_lite_mapper import ModelLiteMapper
from app.core.db.models.model_task import ModelTask
from app.core.settings import get_settings
from app.schemas.requests import model
from app.utils.kfp_client_manager import kfp_client

SETTINGS = get_settings()


class ModelService:
    def lite_model_trt(
        db: Session, model_create_form: model.LiteModelForm
    ) -> dict[str, any]:
        """
        trt를 적용한 모델 경량화
        """
        trt_docker_image_path = ModelLiteMapper.Bert_TRT.value

        # 작업 완료시 server에 요청을 보낼 url
        response_server_url = SETTINGS.SERVER_IP

        uuidV4 = uuid4()
        uuid_str = str(uuidV4)

        # 사용할 컨테이너 정의 및 설정 추가
        @dsl.container_component
        def trt_workflow():
            return dsl.ContainerSpec(
                # 사용할 도커이미지의 주소 및 태그
                image=trt_docker_image_path,
                # 실행할 커맨드
                command=[
                    "pipenv",
                    "run",
                    "python",
                    "main.py",
                ],
                # 필요한 변수들 정의 (string으로 정의)
                args=[
                    "--model_name",
                    model_create_form.name,
                    "--model_run_id",
                    model_create_form.saved_model_run_id,
                    "--model_path",
                    model_create_form.saved_model_path,
                    "--mlflow_tracking_url",
                    model_create_form.mlflow_tracking_url,
                    "--mlflow_s3_endpoint_url",
                    model_create_form.mlflow_s3_endpoint_url,
                    "--server_uuid",
                    uuid_str,
                    "--server_path",
                    f"{response_server_url}/api/v1/tasks",
                ],
            )

        # 파이프라인 정의
        @dsl.pipeline
        def lite_model_trt_test():
            trt_workflow()

        # 정의된 함수로 파이프라인 생성
        run = kfp_client.create_run_from_pipeline_func(
            pipeline_func=lite_model_trt_test,
            namespace=SETTINGS.KUBEFLOW_NAMESPACE,
        )

        kubeflow_experiment_id = run.run_id

        new_task = ModelTask(
            task_uuid=uuid_str,
            model_name=model_create_form.name,
            model_path_input=model_create_form.saved_model_path,
            lite_id=model_create_form.lite_type,
            kubeflow_experiment_id=kubeflow_experiment_id,
        )

        db.add(new_task)
        db.commit()

        return {"task_uuid": uuid_str, "kubeflow_experiment_id": kubeflow_experiment_id}
