import json
import os

import redis


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=None,
)




def submit_task(task_data: dict) -> str:
    """提交异步任务，返回任务 ID。"""
    task_id = redis_client.incr("task_id_counter")

    task = {
        "task_id": str(task_id),
        "status": "pending",
        "data": task_data,
    }

    # 先保存任务状态和数据
    redis_client.hset(
        f"task:{task_id}",
        mapping={
            "status": "pending",
            "data": json.dumps(task_data, ensure_ascii=False),
        },
    )

    # 再将任务放入队列，供消费者处理
    redis_client.lpush("analysis_tasks", json.dumps(task, ensure_ascii=False))

    return str(task_id)



def get_task(task_id: str) -> dict | None:
    """查询任务状态。"""
    task = redis_client.hgetall(f"task:{task_id}")
    return task or None