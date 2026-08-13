# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
from unittest.mock import create_autospec
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.gke_code_executor import GkeCodeExecutor
from kubernetes import client
from kubernetes import config
from kubernetes.client.rest import ApiException
import pydantic
import pytest

try:
  from k8s_agent_sandbox import SandboxClient as RealSandboxClient
  from k8s_agent_sandbox.commands.command_executor import CommandExecutor
  from k8s_agent_sandbox.exceptions import SandboxWarmPoolNotFoundError
  from k8s_agent_sandbox.files.filesystem import Filesystem
  from k8s_agent_sandbox.models import ExecutionResult
  from k8s_agent_sandbox.models import SandboxGatewayConnectionConfig
  from k8s_agent_sandbox.sandbox import Sandbox

  _HAS_AGENT_SANDBOX = True
except ImportError:
  _HAS_AGENT_SANDBOX = False

# The sandbox tests autospec the installed client, so they cannot run without
# it. Skipping only them keeps the job-mode tests running everywhere.
requires_agent_sandbox = pytest.mark.skipif(
    not _HAS_AGENT_SANDBOX, reason="k8s-agent-sandbox is not installed"
)


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Fixture for a mock InvocationContext."""
  mock = MagicMock(spec=InvocationContext)
  mock.invocation_id = "test-invocation-123"
  return mock


@pytest.fixture(autouse=True)
def mock_k8s_config():
  """Fixture for auto-mocking Kubernetes config loading."""
  with patch(
      "google.adk.code_executors.gke_code_executor.config"
  ) as mock_config:
    # Simulate fallback from in-cluster to kubeconfig
    mock_config.ConfigException = config.ConfigException
    mock_config.load_incluster_config.side_effect = config.ConfigException
    yield mock_config


@pytest.fixture
def mock_k8s_clients():
  """Fixture for mock Kubernetes API clients."""
  with patch(
      "google.adk.code_executors.gke_code_executor.client"
  ) as mock_client_class:
    mock_batch_v1 = MagicMock(spec=client.BatchV1Api)
    mock_core_v1 = MagicMock(spec=client.CoreV1Api)
    mock_client_class.BatchV1Api.return_value = mock_batch_v1
    mock_client_class.CoreV1Api.return_value = mock_core_v1
    yield {
        "batch_v1": mock_batch_v1,
        "core_v1": mock_core_v1,
    }


@pytest.fixture
def mock_sandbox_client():
  """Patches SandboxClient with a mock autospecced from the real class.

  `create_autospec` on `Sandbox` leaves `commands` and `files` as plain
  `MagicMock`s, because both are properties, so this fixture specs the two
  nested engines itself. Without that, a call with a signature the real
  library rejects still passes.
  """
  sandbox = create_autospec(Sandbox, instance=True)
  sandbox.claim_name = "sandbox-claim-abc123"
  sandbox.commands = create_autospec(CommandExecutor, instance=True)
  sandbox.files = create_autospec(Filesystem, instance=True)
  sandbox.commands.run.return_value = ExecutionResult(
      stdout="sandbox stdout", stderr="", exit_code=0
  )
  client_class = create_autospec(RealSandboxClient)
  client_class.return_value.create_sandbox.return_value = sandbox
  with patch(
      "google.adk.code_executors.gke_code_executor.SandboxClient", client_class
  ):
    yield {
        "class": client_class,
        "client": client_class.return_value,
        "sandbox": sandbox,
    }


class TestGkeCodeExecutor:
  """Unit tests for the GkeCodeExecutor."""

  def test_init_defaults(self):
    """Tests that the executor initializes with correct default values."""
    executor = GkeCodeExecutor()
    assert executor.namespace == "default"
    assert executor.image == "python:3.11-slim"
    assert executor.timeout_seconds == 300
    assert executor.cpu_requested == "200m"
    assert executor.mem_limit == "512Mi"
    assert executor.executor_type == "job"

  @requires_agent_sandbox
  def test_init_with_overrides(self, mock_sandbox_client):
    """Tests that class attributes can be overridden at instantiation."""
    executor = GkeCodeExecutor(
        namespace="test-ns",
        image="custom-python:latest",
        timeout_seconds=60,
        cpu_limit="1000m",
        executor_type="sandbox",
    )
    assert executor.namespace == "test-ns"
    assert executor.image == "custom-python:latest"
    assert executor.timeout_seconds == 60
    assert executor.cpu_limit == "1000m"
    assert executor.executor_type == "sandbox"
    assert executor.sandbox_warmpool == "python-sandbox-warmpool"
    assert executor.sandbox_template is None

  def test_init_backward_compatibility(self):
    """Tests that the executor can be initialized with positional arguments."""
    executor = GkeCodeExecutor(
        "/path/to/kubeconfig",
        "test-context",
        namespace="test-ns",
        image="test-image",
        timeout_seconds=100,
        executor_type="job",
        cpu_requested="100m",
        mem_requested="128Mi",
        cpu_limit="200m",
        mem_limit="256Mi",
    )
    assert executor.namespace == "test-ns"
    assert executor.image == "test-image"
    assert executor.timeout_seconds == 100
    assert executor.executor_type == "job"
    assert executor.cpu_requested == "100m"
    assert executor.mem_requested == "128Mi"
    assert executor.cpu_limit == "200m"
    assert executor.mem_limit == "256Mi"
    assert executor.kubeconfig_path == "/path/to/kubeconfig"
    assert executor.kubeconfig_context == "test-context"

  def test_init_partial_positional_args(self):
    """Tests initialization with partial positional arguments."""
    executor = GkeCodeExecutor("/path/to/kubeconfig")
    assert executor.kubeconfig_path == "/path/to/kubeconfig"
    assert executor.kubeconfig_context is None

  def test_init_mixed_args(self):
    """Tests initialization with mixed positional and keyword arguments."""
    executor = GkeCodeExecutor(
        "/path/to/kubeconfig",
        kubeconfig_context="test-context",
        namespace="test-ns",
    )
    assert executor.kubeconfig_path == "/path/to/kubeconfig"

  @pytest.mark.parametrize("timeout", [0, -1, None])
  def test_non_positive_timeout_is_rejected(self, timeout):
    """A timeout of 0 or None would mean no bound, so it is refused up front."""
    with pytest.raises(pydantic.ValidationError):
      GkeCodeExecutor(timeout_seconds=timeout)

  def test_init_sandbox_missing_dependency(self):
    """Tests that init raises ImportError if k8s-agent-sandbox is missing."""
    with patch(
        "google.adk.code_executors.gke_code_executor.SandboxClient", None
    ):
      with pytest.raises(ImportError, match="k8s-agent-sandbox not found"):
        GkeCodeExecutor(executor_type="sandbox")

        GkeCodeExecutor(executor_type="sandbox")

  @patch("google.adk.code_executors.gke_code_executor.Watch")
  def test_execute_code_success(
      self,
      mock_watch,
      mock_k8s_clients,
      mock_invocation_context,
  ):
    """Tests the happy path for successful code execution."""
    # Setup Mocks
    mock_job = MagicMock()
    mock_job.status.succeeded = True
    mock_job.status.failed = None
    mock_watch.return_value.stream.return_value = [{"object": mock_job}]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [MagicMock()]
    mock_pod_list.items[0].metadata.name = "test-pod-name"
    mock_k8s_clients["core_v1"].list_namespaced_pod.return_value = mock_pod_list
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.return_value = (
        "hello world"
    )

    # Execute
    executor = GkeCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert
    assert result.stdout == "hello world"
    assert result.stderr == ""
    mock_k8s_clients[
        "core_v1"
    ].create_namespaced_config_map.assert_called_once()
    mock_k8s_clients["batch_v1"].create_namespaced_job.assert_called_once()
    mock_k8s_clients["core_v1"].patch_namespaced_config_map.assert_called_once()
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.assert_called_once()

  @patch("google.adk.code_executors.gke_code_executor.Watch")
  def test_execute_code_job_failed(
      self,
      mock_watch,
      mock_k8s_clients,
      mock_invocation_context,
  ):
    """Tests the path where the Kubernetes Job fails."""
    mock_job = MagicMock()
    mock_job.status.succeeded = None
    mock_job.status.failed = True
    mock_watch.return_value.stream.return_value = [{"object": mock_job}]
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.return_value = (
        "Traceback...\nValueError: failure"
    )

    executor = GkeCodeExecutor()
    result = executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="fail")
    )

    assert result.stdout == ""
    assert "Job failed. Logs:" in result.stderr
    assert "ValueError: failure" in result.stderr

  def test_execute_code_api_exception(
      self, mock_k8s_clients, mock_invocation_context
  ):
    """Tests handling of an ApiException from the K8s client."""
    mock_k8s_clients["core_v1"].create_namespaced_config_map.side_effect = (
        ApiException(reason="Test API Error")
    )
    executor = GkeCodeExecutor()
    result = executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="...")
    )

    assert result.stdout == ""
    assert "Kubernetes API error: Test API Error" in result.stderr

  @patch("google.adk.code_executors.gke_code_executor.Watch")
  def test_execute_code_timeout(
      self,
      mock_watch,
      mock_k8s_clients,
      mock_invocation_context,
  ):
    """Tests the case where the job watch times out."""
    mock_watch.return_value.stream.return_value = (
        []
    )  # Empty stream simulates timeout
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.return_value = (
        "Still running..."
    )

    executor = GkeCodeExecutor(timeout_seconds=1)
    result = executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="...")
    )

    assert result.stdout == ""
    assert "Executor timed out" in result.stderr
    assert "did not complete within 1s" in result.stderr
    assert "Pod Logs:\nStill running..." in result.stderr

  def test_create_job_manifest_structure(self, mock_invocation_context):
    """Tests the correctness of the generated Job manifest."""
    executor = GkeCodeExecutor(namespace="test-ns", image="test-img:v1")
    job = executor._create_job_manifest(
        "test-job", "test-cm", mock_invocation_context
    )

    # Check top-level properties
    assert isinstance(job, client.V1Job)
    assert job.api_version == "batch/v1"
    assert job.kind == "Job"
    assert job.metadata.name == "test-job"
    assert job.spec.backoff_limit == 0
    assert job.spec.ttl_seconds_after_finished == 600

    # Check pod template properties
    pod_spec = job.spec.template.spec
    assert pod_spec.restart_policy == "Never"
    assert pod_spec.automount_service_account_token is False
    assert pod_spec.runtime_class_name == "gvisor"
    assert len(pod_spec.tolerations) == 1
    assert pod_spec.tolerations[0].value == "gvisor"
    assert len(pod_spec.volumes) == 1
    assert pod_spec.volumes[0].name == "code-volume"
    assert pod_spec.volumes[0].config_map.name == "test-cm"

    # Check container properties
    container = pod_spec.containers[0]
    assert container.name == "code-runner"
    assert container.image == "test-img:v1"
    assert container.command == ["python3", "/app/code.py"]

    # Check security context
    sec_context = container.security_context
    assert sec_context.run_as_non_root is True
    assert sec_context.run_as_user == 1001
    assert sec_context.allow_privilege_escalation is False
    assert sec_context.read_only_root_filesystem is True
    assert sec_context.capabilities.drop == ["ALL"]

  @requires_agent_sandbox
  def test_execute_code_forks_to_sandbox(
      self,
      mock_sandbox_client,
      mock_invocation_context,
      mock_k8s_clients,
  ):
    """Tests that execute_code provisions a sandbox from the warm pool."""
    executor = GkeCodeExecutor(executor_type="sandbox")
    code_input = CodeExecutionInput(code='print("sandbox")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "sandbox stdout"
    mock_sandbox_client["client"].create_sandbox.assert_called_once_with(
        warmpool="python-sandbox-warmpool",
        namespace="default",
        sandbox_ready_timeout=300,
    )
    mock_k8s_clients["batch_v1"].create_namespaced_job.assert_not_called()

  @requires_agent_sandbox
  def test_execute_in_sandbox_returns_stderr(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that stderr from the sandbox run is propagated to the result."""
    mock_sandbox_client["sandbox"].commands.run.return_value = ExecutionResult(
        stdout="", stderr="oops\n", exit_code=1
    )
    executor = GkeCodeExecutor(executor_type="sandbox")
    code_input = CodeExecutionInput(
        code="import sys; print('oops', file=sys.stderr)"
    )

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert result.stderr == "oops\n"
    mock_sandbox_client["sandbox"].files.write.assert_called_once_with(
        "script.py", code_input.code, timeout=300
    )
    mock_sandbox_client["sandbox"].commands.run.assert_called_once_with(
        "python3 script.py", timeout=300
    )

  @requires_agent_sandbox
  def test_execute_code_sandbox_connection_error(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that a non-RuntimeError sandbox failure propagates unchanged."""
    mock_sandbox_client["client"].create_sandbox.side_effect = ValueError(
        "Connection failed"
    )
    executor = GkeCodeExecutor(executor_type="sandbox")
    code_input = CodeExecutionInput(code='print("sandbox")')

    with pytest.raises(ValueError, match="Connection failed"):
      executor.execute_code(mock_invocation_context, code_input)

  @requires_agent_sandbox
  def test_execute_code_sandbox_runtime_error(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that a SandboxError is rewrapped as an infrastructure error."""
    mock_sandbox_client["client"].create_sandbox.side_effect = (
        SandboxWarmPoolNotFoundError("Gateway not found")
    )
    executor = GkeCodeExecutor(executor_type="sandbox")
    code_input = CodeExecutionInput(code='print("sandbox")')

    with pytest.raises(
        RuntimeError, match="Sandbox infrastructure error: Gateway not found"
    ):
      executor.execute_code(mock_invocation_context, code_input)

  @requires_agent_sandbox
  def test_execute_code_sandbox_timeout_error(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that a TimeoutError is returned as a result, not raised."""
    mock_sandbox_client["client"].create_sandbox.side_effect = TimeoutError(
        "Execution timed out"
    )
    executor = GkeCodeExecutor(executor_type="sandbox")
    code_input = CodeExecutionInput(code='print("sandbox")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert "Sandbox timed out: Execution timed out" in result.stderr

  @requires_agent_sandbox
  @patch("google.adk.code_executors.gke_code_executor.Watch")
  def test_execute_code_forks_to_job(
      self,
      mock_watch,
      mock_sandbox_client,
      mock_invocation_context,
      mock_k8s_clients,
  ):
    """Tests that execute_code uses K8s Job when executor_type='job'."""
    mock_job = MagicMock()
    mock_job.status.succeeded = True
    mock_watch.return_value.stream.return_value = [{"object": mock_job}]

    mock_pod = MagicMock()
    mock_pod.metadata.name = "pod-1"
    mock_k8s_clients["core_v1"].list_namespaced_pod.return_value.items = [
        mock_pod
    ]
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.return_value = (
        "job stdout"
    )

    executor = GkeCodeExecutor(executor_type="job")
    code_input = CodeExecutionInput(code='print("job")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "job stdout"
    mock_k8s_clients["batch_v1"].create_namespaced_job.assert_called_once()
    mock_sandbox_client["class"].assert_not_called()

  @requires_agent_sandbox
  def test_execute_in_sandbox_terminates_on_success(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that a successful run still deletes the sandbox claim."""
    executor = GkeCodeExecutor(executor_type="sandbox")

    executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("sandbox")')
    )

    mock_sandbox_client["sandbox"].terminate.assert_called_once_with()

  @requires_agent_sandbox
  def test_execute_in_sandbox_terminates_on_failure(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that a failed run still deletes the sandbox claim."""
    mock_sandbox_client["sandbox"].commands.run.side_effect = ValueError(
        "run exploded"
    )
    executor = GkeCodeExecutor(executor_type="sandbox")

    with pytest.raises(ValueError, match="run exploded"):
      executor.execute_code(
          mock_invocation_context, CodeExecutionInput(code='print("sandbox")')
      )

    mock_sandbox_client["sandbox"].terminate.assert_called_once_with()

  @requires_agent_sandbox
  def test_execute_in_sandbox_swallows_terminate_failure(
      self,
      mock_sandbox_client,
      mock_invocation_context,
      caplog,
  ):
    """Tests that a cleanup failure warns and keeps the execution result."""
    mock_sandbox_client["sandbox"].terminate.side_effect = ValueError(
        "claim already gone"
    )
    executor = GkeCodeExecutor(executor_type="sandbox")

    with caplog.at_level(logging.WARNING):
      result = executor.execute_code(
          mock_invocation_context, CodeExecutionInput(code='print("sandbox")')
      )

    assert result.stdout == "sandbox stdout"
    assert "sandbox-claim-abc123" in caplog.text
    assert "claim already gone" in caplog.text

  @requires_agent_sandbox
  def test_execute_in_sandbox_uses_gateway_connection_config(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that a named gateway reaches the client as a gateway config."""
    executor = GkeCodeExecutor(
        executor_type="sandbox", sandbox_gateway_name="external-http-gateway"
    )

    executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("sandbox")')
    )

    mock_sandbox_client["class"].assert_called_once_with(
        connection_config=SandboxGatewayConnectionConfig(
            gateway_name="external-http-gateway"
        )
    )

  @requires_agent_sandbox
  def test_execute_in_sandbox_defaults_to_local_tunnel(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that an unset gateway leaves the client on its own default."""
    executor = GkeCodeExecutor(executor_type="sandbox")

    executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("sandbox")')
    )

    mock_sandbox_client["class"].assert_called_once_with(connection_config=None)

  @requires_agent_sandbox
  def test_execute_in_sandbox_applies_timeout_seconds(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that timeout_seconds bounds readiness, upload and execution."""
    executor = GkeCodeExecutor(executor_type="sandbox", timeout_seconds=600)

    executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("sandbox")')
    )

    assert (
        mock_sandbox_client["client"].create_sandbox.call_args.kwargs[
            "sandbox_ready_timeout"
        ]
        == 600
    )
    assert (
        mock_sandbox_client["sandbox"].files.write.call_args.kwargs["timeout"]
        == 600
    )
    assert (
        mock_sandbox_client["sandbox"].commands.run.call_args.kwargs["timeout"]
        == 600
    )

  @requires_agent_sandbox
  def test_sandbox_run_uses_default_timeout(
      self,
      mock_sandbox_client,
      mock_invocation_context,
  ):
    """Tests that the field default, not the client's own 60s, is sent."""
    executor = GkeCodeExecutor(executor_type="sandbox")

    executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("sandbox")')
    )

    mock_sandbox_client["sandbox"].commands.run.assert_called_once_with(
        "python3 script.py", timeout=300
    )

  @requires_agent_sandbox
  def test_sandbox_template_is_deprecated_alias(self, mock_sandbox_client):
    """Tests that a legacy sandbox_template warns and names the warm pool."""
    with pytest.warns(DeprecationWarning, match="sandbox_warmpool"):
      executor = GkeCodeExecutor(
          executor_type="sandbox", sandbox_template="legacy-pool"
      )

    assert executor.sandbox_warmpool == "legacy-pool"

  @requires_agent_sandbox
  def test_sandbox_warmpool_wins_over_template(self, mock_sandbox_client):
    """Tests that an explicit sandbox_warmpool overrides the legacy alias."""
    with pytest.warns(DeprecationWarning, match="sandbox_warmpool"):
      executor = GkeCodeExecutor(
          executor_type="sandbox",
          sandbox_template="legacy-pool",
          sandbox_warmpool="python-sandbox-warmpool",
      )

    assert executor.sandbox_warmpool == "python-sandbox-warmpool"


@requires_agent_sandbox
def test_sandbox_client_api_shape():
  """Pins the k8s-agent-sandbox surface the executor is written against.

  This fails the day the dependency floor moves past a release that renames
  or drops one of these members, which is how the previous client generation
  went unnoticed.
  """
  create_sandbox = inspect.signature(RealSandboxClient.create_sandbox)
  assert "warmpool" in create_sandbox.parameters
  assert (
      "connection_config"
      in inspect.signature(RealSandboxClient.__init__).parameters
  )
  assert "claim_name" in inspect.signature(Sandbox.__init__).parameters
  assert "timeout" in inspect.signature(CommandExecutor.run).parameters
  assert "timeout" in inspect.signature(Filesystem.write).parameters
  for member in ("commands", "files", "terminate"):
    assert hasattr(Sandbox, member)
  assert {"stdout", "stderr"} <= set(ExecutionResult.model_fields)
  assert "gateway_name" in SandboxGatewayConnectionConfig.model_fields
