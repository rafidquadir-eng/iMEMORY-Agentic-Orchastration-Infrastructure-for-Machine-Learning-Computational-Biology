"""Round-trip: plan -> assign -> execute -> return. No network required."""
from agents.a2a import A2AHandoff, A2AMessage


def test_assign_and_return_roundtrip():
    task = A2AHandoff.assign_task(
        description="nominate targets",
        code="run nomination",
        expected_outputs=["results.json"],
    )
    assert task.sender_id == "planning_agent"
    assert task.receiver_id == "execution_agent"
    assert task.message_type == "task_assignment"

    result = A2AHandoff.return_results(task, results={"nominated_targets": [("A", 0.78)]},
                                       audit_passed=True)
    assert result.message_type == "result_return"
    assert result.conversation_id == task.conversation_id      # lineage preserved
    assert result.parent_message_id == task.message_id
    assert result.payload["audit_passed"] is True


def test_log_record_is_serializable():
    msg = A2AMessage(
        sender_id="planning_agent",
        receiver_id="execution_agent",
        message_type="task_assignment",
        task_description="x",
        conversation_id="conv-1",
    )
    rec = msg.to_log_record()
    assert rec["conversation_id"] == "conv-1"
    assert "payload_keys" in rec
