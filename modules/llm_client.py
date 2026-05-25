import json
import logging
import re
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests
from requests.exceptions import HTTPError, RequestException, Timeout

from config import Config

logger = logging.getLogger(__name__)

class ModelUnavailableError(Exception):
    pass


class ModelMemoryError(ModelUnavailableError):
    pass


class InvalidLLMOutputError(ModelUnavailableError):
    def __init__(self, message: str, raw_text: str = ''):
        super().__init__(message)
        self.raw_text = raw_text


def _request_json(path: str, timeout: int = None) -> Dict[str, Any]:
    url = urljoin(Config.LLM_SERVER_URL.rstrip('/') + '/', path.lstrip('/'))
    try:
        response = requests.get(url, timeout=timeout or Config.LLM_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except RequestException as exc:
        logger.error("LLM health request failed: %s", exc)
        raise ModelUnavailableError("LLM server is unavailable") from exc


def _post_json(path: str, payload: dict, timeout: int = None) -> Dict[str, Any]:
    url = urljoin(Config.LLM_SERVER_URL.rstrip('/') + '/', path.lstrip('/'))
    try:
        response = requests.post(url, json=payload, timeout=timeout or Config.LLM_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Timeout as exc:
        logger.error("LLM request timed out after %s seconds: %s", timeout or Config.LLM_TIMEOUT, url)
        raise ModelUnavailableError(
            "LLM inference timed out after %s seconds. Increase LLM_GENERATION_TIMEOUT or use a faster model."
            % (timeout or Config.LLM_TIMEOUT)
        ) from exc
    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = exc.response.text[:500] if exc.response is not None else ""
        logger.error("LLM request returned HTTP %s: %s", status, body)
        if "requires more system memory" in body.lower():
            raise ModelMemoryError(
                "Selected LLM cannot fit in available memory. Ollama response: %s"
                % body
            ) from exc
        raise ModelUnavailableError("LLM inference failed with HTTP %s: %s" % (status, body)) from exc
    except RequestException as exc:
        logger.error("LLM request failed: %s", exc)
        raise ModelUnavailableError("LLM inference failed: %s" % str(exc)) from exc


def _extract_model_list(data: Any) -> List[Any]:
    if isinstance(data, dict):
        if data.get('object') == 'list' and isinstance(data.get('data'), list):
            return data['data']
        if isinstance(data.get('models'), list):
            return data['models']
        if isinstance(data.get('data'), list):
            return data['data']
        return []
    if isinstance(data, list):
        return data
    return []


def _find_model_id(models: List[Any], model_name: str) -> str:
    for item in models:
        if isinstance(item, str):
            if item == model_name or item.startswith(model_name):
                return item
        elif isinstance(item, dict):
            if item.get('id') == model_name or item.get('name') == model_name:
                return item.get('id') or item.get('name')
    for item in models:
        model_id = None
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            model_id = item.get('id') or item.get('name')
        if model_id and model_id.startswith(model_name):
            return model_id
    raise ModelUnavailableError(f"LLM model '{model_name}' is not present in the server model list")


def check_model_available() -> str:
    try:
        models_response = _request_json('/v1/models')
        models = _extract_model_list(models_response)
        model_id = _find_model_id(models, Config.LLM_MODEL_NAME)
        logger.info("LLM model '%s' resolved to '%s'", Config.LLM_MODEL_NAME, model_id)
        return model_id
    except ModelUnavailableError as exc:
        logger.warning("Model list endpoint failed: %s", exc)

    try:
        status = _request_json('/v1/status')
        if isinstance(status, dict) and status.get('status') in {'ok', 'healthy', 'available'}:
            logger.info("LLM server is healthy")
            return Config.LLM_MODEL_NAME
    except ModelUnavailableError:
        pass

    raise ModelUnavailableError(
        "LLM model is not available. Ensure the local server is running and the model name is correct."
    )


def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE)
    return text.strip()


def _extract_response_text(payload: dict) -> str:
    if isinstance(payload, dict) and payload.get('choices'):
        choice = payload['choices'][0]
        if isinstance(choice, dict):
            message = choice.get('message')
            if isinstance(message, dict):
                return message.get('content', '')
            return choice.get('text', '')
    text = payload.get('response') or payload.get('output') or payload.get('text')
    if isinstance(text, list):
        text = ' '.join(text)
    if text is None:
        raise ModelUnavailableError('LLM response did not include a valid text payload')
    return text


def _normalize_parsed_response(data: dict) -> dict:
    """Normalize numeric scores and preserve comment fields from parsed JSON."""
    result = {}
    for key, value in data.items():
        if key in ('communication', 'confidence', 'content', 'structure'):
            if isinstance(value, (int, float)):
                result[key] = max(1.0, min(10.0, value))
            elif isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, (int, float)):
                        result[key] = max(1.0, min(10.0, sub_value))
                        break
            else:
                # preserve invalid values to detect missing or malformed scores later
                result[key] = value
        else:
            result[key] = value

    reason = result.get('reason') if 'reason' in result else data.get('reason')
    if isinstance(reason, str):
        result['reason'] = reason.strip()
    elif isinstance(reason, dict):
        flattened = next((str(v).strip() for v in reason.values() if isinstance(v, str) and v.strip()), None)
        if flattened:
            result['reason'] = flattened
    return result


def _extract_partial_scores(text: str) -> dict:
    """Extract numeric scores from partially truncated JSON using regex, tolerating nested structures."""
    result = {}
    patterns = {
        'communication': r'"communication"\s*:\s*[^,}]*?(\d+)',
        'confidence': r'"confidence"\s*:\s*[^,}]*?(\d+)',
        'content': r'"content"\s*:\s*[^,}]*?(\d+)',
        'structure': r'"structure"\s*:\s*[^,}]*?(\d+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                score = int(match.group(1))
                result[key] = max(1.0, min(10.0, float(score)))
            except (ValueError, IndexError):
                pass
    if result:
        result['reason'] = 'Extracted from model response.'
        logger.info('Regex extraction succeeded for keys: %s', list(result.keys()))
    return result


def _extract_reason_text(text: str) -> str:
    reason_pattern = re.search(r'"reason"\s*:\s*"([^"]*)"', text, re.IGNORECASE | re.DOTALL)
    if reason_pattern:
        return reason_pattern.group(1).strip()
    reason_obj = re.search(r'"reason"\s*:\s*\{([^}]*)\}', text, re.IGNORECASE | re.DOTALL)
    if reason_obj:
        candidate = re.findall(r'"[^"]+"\s*:\s*"([^"]*)"', reason_obj.group(1))
        if candidate:
            return ' '.join([c.strip() for c in candidate if c.strip()])
    return ''


def _clean_json_like(text: str) -> str:
    text = text.strip()
    text = re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'([\{,\s])([a-zA-Z_][a-zA-Z0-9_]*)(\s*):', r'\1"\2"\3:', text)

    def replace_single_quotes(match):
        inner = match.group(1)
        if '"' in inner:
            return match.group(0)
        return '"' + inner.replace('"', '\\"') + '"'

    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", replace_single_quotes, text)
    return text


def _extract_json_text(text: str) -> str:
    text = _strip_code_fences(text)
    candidates = []
    stack = []
    start = None

    for index, char in enumerate(text):
        if char == '{':
            if start is None:
                start = index
            stack.append(char)
        elif char == '}' and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:index + 1])
                start = None

    if candidates:
        return candidates[0]
    return text


def _parse_json_text(text: str) -> Dict[str, Any]:
    raw_text = text
    candidate = _extract_json_text(text)

    def try_parse(s: str) -> Dict[str, Any]:
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError('Parsed LLM output is not a JSON object')
        return _normalize_parsed_response(parsed)

    try:
        return try_parse(candidate)
    except (json.JSONDecodeError, ValueError):
        cleaned = _clean_json_like(candidate)
        try:
            return try_parse(cleaned)
        except (json.JSONDecodeError, ValueError):
            brace_positions = [i for i, c in enumerate(raw_text) if c == '{']
            for start in brace_positions:
                depth = 0
                for index in range(start, len(raw_text)):
                    if raw_text[index] == '{':
                        depth += 1
                    elif raw_text[index] == '}':
                        depth -= 1
                        if depth == 0:
                            snippet = raw_text[start:index + 1]
                            cleaned_snippet = _clean_json_like(snippet)
                            try:
                                return try_parse(cleaned_snippet)
                            except (json.JSONDecodeError, ValueError):
                                continue
        logger.error('Failed to parse any JSON object from LLM output: %s', raw_text[:500])
        raise InvalidLLMOutputError(
            'LLM returned invalid JSON output; raw output is not parseable JSON.',
            raw_text=raw_text[:1000]
        )


def generate_evaluation(prompt: str) -> Dict[str, Any]:
    model_id = check_model_available()

    payload = {
        'model': model_id,
        'messages': [
            {'role': 'system', 'content': 'You are a strict technical interviewer. Use only provided evidence and respond with valid JSON only.'},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0.0,
        'max_tokens': 1000,
        'top_p': 1.0,
    }

    result = _post_json('/v1/chat/completions', payload, timeout=Config.LLM_GENERATION_TIMEOUT)
    response_text = _extract_response_text(result)
    try:
        parsed = _parse_json_text(response_text)
    except InvalidLLMOutputError as exc:
        logger.error('LLM output invalid JSON: %s', exc.raw_text)
        raise
    if isinstance(parsed, dict):
        parsed['raw_llm_response'] = response_text
        if 'reason' not in parsed:
            extracted_reason = _extract_reason_text(response_text)
            if extracted_reason:
                parsed['reason'] = extracted_reason
    return parsed
