# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from types import SimpleNamespace

from google.adk.events.event import Event
from google.adk.memory.vertex_ai_rag_memory_service import _build_source_display_name
from google.adk.memory.vertex_ai_rag_memory_service import _SOURCE_DISPLAY_NAME_PREFIX
from google.adk.memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


def _rag_context(source_display_name: str, text: str) -> SimpleNamespace:
  return SimpleNamespace(
      source_display_name=source_display_name,
      text=json.dumps({"author": "user", "timestamp": 1, "text": text}),
  )


@pytest.mark.asyncio
async def test_search_memory_rejects_ambiguous_legacy_display_names(mocker):
  """Ensures dotted user IDs cannot match another user's legacy memory."""
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")
  fake_filter = mocker.Mock()
  fake_config = mocker.Mock()
  fake_retrieval_query = mocker.Mock(
      return_value=SimpleNamespace(
          contexts=SimpleNamespace(
              contexts=[
                  _rag_context(
                      "demo.alice.smith.session_secret",
                      "SECRET_FROM_ALICE_SMITH",
                  ),
                  _rag_context(
                      _build_source_display_name("demo", "alice", "session_ok"),
                      "NORMAL_ALICE_MEMORY",
                  ),
                  _rag_context(
                      "demo.alice.legacy_session",
                      "LEGACY_ALICE_MEMORY",
                  ),
              ]
          )
      )
  )

  fake_rag = SimpleNamespace(
      Filter=fake_filter,
      RagRetrievalConfig=fake_config,
      retrieval_query=fake_retrieval_query,
  )
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)

  response = await memory_service.search_memory(
      app_name="demo", user_id="alice", query="secret"
  )

  fake_filter.assert_called_once_with(
      vector_distance_threshold=10.0,
      metadata_filter=(
          "regexp_contains(DisplayName,"
          " '^(adk\\-memory\\-v1\\.ZGVtbw\\.YWxpY2U\\.|demo\\.alice\\.)')"
      ),
  )
  fake_config.assert_called_once_with(
      top_k=None,
      filter=fake_filter.return_value,
  )
  fake_retrieval_query.assert_called_once_with(
      text="secret",
      rag_resources=memory_service._vertex_rag_store.rag_resources,
      rag_corpora=memory_service._vertex_rag_store.rag_corpora,
      rag_retrieval_config=fake_config.return_value,
  )

  texts = [memory.content.parts[0].text for memory in response.memories]
  assert texts == ["NORMAL_ALICE_MEMORY", "LEGACY_ALICE_MEMORY"]


@pytest.mark.asyncio
async def test_add_and_search_memory_uses_unambiguous_display_names(mocker):
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")
  upload_file = mocker.Mock()
  fake_filter = mocker.Mock()
  fake_config = mocker.Mock()
  fake_rag = SimpleNamespace(
      upload_file=upload_file,
      Filter=fake_filter,
      RagRetrievalConfig=fake_config,
  )
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)

  await memory_service.add_session_to_memory(
      Session(
          app_name="demo.app",
          user_id="alice.smith",
          id="session.secret",
          last_update_time=1,
          events=[
              Event(
                  id="event-1",
                  author="user",
                  timestamp=1,
                  content=types.Content(
                      parts=[types.Part(text="sensitive memory")]
                  ),
              )
          ],
      )
  )

  display_name = upload_file.call_args.kwargs["display_name"]
  assert display_name.startswith(_SOURCE_DISPLAY_NAME_PREFIX)
  assert display_name != "demo.app.alice.smith.session.secret"

  fake_retrieval_query = mocker.Mock(
      return_value=SimpleNamespace(
          contexts=SimpleNamespace(
              contexts=[_rag_context(display_name, "sensitive memory")]
          )
      )
  )
  fake_rag.retrieval_query = fake_retrieval_query

  response = await memory_service.search_memory(
      app_name="demo.app", user_id="alice.smith", query="sensitive"
  )

  fake_filter.assert_called_once_with(
      vector_distance_threshold=10.0,
      metadata_filter=(
          "regexp_contains(DisplayName,"
          " '^(adk\\-memory\\-v1\\.ZGVtby5hcHA\\.YWxpY2Uuc21pdGg\\.|demo\\.app\\.alice\\.smith\\.)')"
      ),
  )
  fake_config.assert_called_once_with(
      top_k=None,
      filter=fake_filter.return_value,
  )
  fake_retrieval_query.assert_called_once_with(
      text="sensitive",
      rag_resources=memory_service._vertex_rag_store.rag_resources,
      rag_corpora=memory_service._vertex_rag_store.rag_corpora,
      rag_retrieval_config=fake_config.return_value,
  )

  assert [memory.content.parts[0].text for memory in response.memories] == [
      "sensitive memory"
  ]
