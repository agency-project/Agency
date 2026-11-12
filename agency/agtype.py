from __future__ import annotations
import base64
import mimetypes
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, get_args, get_origin
from .utils.agutil import _looks_like_path

if TYPE_CHECKING:
    from .sandbox.agsandbox import agSandbox


class agtype:
    """Base class for typed agdata field_name values.

    Subclass to add custom serialization, sandbox I/O, system prompts, or
    cleanup logic for a field_name type.  The default implementations pass plain
    Python values through unchanged.

    Class-level use
    ---------------
    An ``agtype`` subclass is used as a *type marker* in agskill schemas::

        skill = agskill(
            name="write",
            prompt="...",
            input_schema=agdata(theme=str, background=agfile),
            output_schema=agdata(report=agfile),
        )

    The class object itself (not an instance) is stored in the schema agdata.
    All interface methods are therefore ``@classmethod`` so they can be called
    on the class without instantiation.

    Interface methods
    -----------------
    schema_type() -> str
        Human-readable type label shown in the JSON format type_hint in the system
        prompt.  Default: ``"str"``.

    needs_sandbox() -> bool
        True if this type requires sandbox filesystem access during
        prepare/recover.  Default: ``False``.

    prepare(value, sandbox, skill_name, field_name) -> tuple[str, list[str]]
        Called before the skill's ReAct loop.  ``value`` is the raw Python
        value from agdata.  Returns (transformed_value, paths_to_cleanup).
        ``transformed_value`` replaces the field_name in the JSON sent to the LLM.
        Default: returns (value, []).

    recover(value, sandbox) -> tuple[str, list[str]]
        Called after the skill's ReAct loop on output fields.  ``value`` is
        whatever the LLM returned for this field_name.  Returns
        (recovered_value, paths_to_cleanup).  Default: returns (value, []).

    extra_input_prompt(field_name) -> str
        Additional instruction line added to the system prompt for this input
        field_name.  Return an empty string to add nothing.  Default: ``""``.

    extra_output_prompt(field_name, skill_name) -> str
        Additional instruction line added to the system prompt for this output
        field_name.  Return an empty string to add nothing.  Default: ``""``.

    build_content_prompt(key, value) -> tuple[str | None, list[dict]]
        Called by build_user_content to let the type contribute to the user
        message content array.  Returns (placeholder, content_blocks).
        placeholder: if not None, replaces the field value in the JSON text
        portion so the LLM doesn't see the raw value (e.g. a base64 blob).
        content_blocks: extra multimodal content dicts to append (e.g. image_url).
        Default: (None, []) — no substitution, no extra blocks.
    """

    @classmethod
    def schema_type(cls) -> str:
        return "str"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[object, list[str]]:
        return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[object, list[str]]:
        return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return ""

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return ""

    @classmethod
    def build_content_prompt(cls, key: str, value: object) -> "tuple[str | None, list[dict]]":
        return None, []

    @classmethod
    def get_return_tool_description(cls, field_name: str) -> str:
        """Description for the return_<field_name> tool itself."""
        return f"Register the '{field_name}' output field_name."

    @classmethod
    def get_return_tool_value_description(cls, field_name: str) -> str:
        """Description for the `value` parameter of return_<field_name>."""
        return f"Value for '{field_name}'"

    @classmethod
    def validate_output(
        cls,
        field_name: str,
        value: object,
        sandbox: "agSandbox",
        exec_timeout: int,
    ) -> "str | None":
        """Validate an output value after the LLM submits it via return_<field_name>.

        Returns an error string to feed back to the LLM, or None if valid.
        Override in subclasses that need sandbox-side checks (agfile, agbinary).
        """
        return None

    @classmethod
    def validate_input_value(cls, value: object) -> "str | None":
        """Validate a raw input value before the skill runs (called from
        agschema.check()/check_field())."""
        return None if isinstance(value, str) else f"must be a string, got {type(value).__name__}"

    @staticmethod
    def walk(type_hint, value, on_leaf):
        """Recursively walk a type type_hint/value pair, calling on_leaf(type_hint, value)
        at every agtype leaf.  Returns (new_value, paths).

        Handles list, dict, and tuple containers at any nesting depth.
        on_leaf must handle its own exceptions and always return (value, []).
        """
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if isinstance(type_hint, type) and issubclass(type_hint, agtype):
            output = on_leaf(type_hint, value)
        elif origin is list and args:
            if not isinstance(value, list):
                output = value, []
            else:
                new_vals, paths = [], []
                for v in value:
                    nv, written = agtype.walk(args[0], v, on_leaf)
                    new_vals.append(nv)
                    paths.extend(written)
                output = new_vals, paths
        elif origin is dict and len(args) == 2:
            if not isinstance(value, dict):
                output = value, []
            else:
                new_vals, paths = {}, []
                for k, v in value.items():
                    nv, written = agtype.walk(args[1], v, on_leaf)
                    new_vals[k] = nv
                    paths.extend(written)
                output = new_vals, paths
        elif origin is tuple and args:
            if not isinstance(value, (list, tuple)):
                output = value, []
            else:
                new_vals, paths = list(value), []
                for i, (type_arg, v) in enumerate(zip(args, value)):
                    nv, written = agtype.walk(type_arg, v, on_leaf)
                    new_vals[i] = nv
                    paths.extend(written)
                output = tuple(new_vals), paths
        else:
            output = value, []

        return output

    @staticmethod
    def in_hint(type_hint) -> bool:
        """Return True if type_hint contains any agtype subclass other than agrawstring,
        at any nesting depth."""
        if isinstance(type_hint, type) and issubclass(type_hint, agtype):
            return not issubclass(type_hint, agrawstring)
        origin = get_origin(type_hint)
        args = get_args(type_hint)
        if origin is list and args:
            return agtype.in_hint(args[0])
        if origin is dict and len(args) == 2:
            return agtype.in_hint(args[1])
        if origin is tuple and args:
            return any(agtype.in_hint(a) for a in args)
        return False

    @staticmethod
    def from_hint(type_hint: object) -> "type[agtype] | None":
        """Return the agtype subclass for a type_hint, handling both T and list[T]."""
        if isinstance(type_hint, type) and issubclass(type_hint, agtype):
            return type_hint
        if get_origin(type_hint) is list:
            args = get_args(type_hint)
            if args and isinstance(args[0], type) and issubclass(args[0], agtype):
                return args[0]
        return None


class agfile(agtype):
    """File-backed agskill schema field_name.

    Input fields
    ------------
    The framework writes the Python string value to a temporary file inside
    the agent's sandbox before the skill runs.  The LLM receives the file
    path and uses the ``read`` tool to access the content.

    Output fields
    -------------
    The agent writes its output to a file and returns the path.  The
    framework reads the file content back and stores it as a plain string
    in the result agdata.  The agent-side file is deleted after the skill
    ends.

    Example::

        design_skill = agskill(
            name="design",
            prompt="Create a story design document.",
            input_schema=agdata(theme=str),
            output_schema=agdata(design_doc=agfile),
        )
    """

    @classmethod
    def schema_type(cls) -> str:
        return "file"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return True

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        path = f"/workspace/inputs/{field_name}{suffix}.txt"
        try:
            sandbox.write_file(path, value)
            return path, [path]
        except Exception as _e:
            print(f"[agtype] WARNING: agfile.prepare failed to write {path}: {_e}")
            return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        try:
            content = sandbox.read_file(value)
            return content, [value]
        except Exception as _e:
            print(f"[agtype] WARNING: agfile.recover failed to read {value}: {_e}")
            return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: the JSON value is a path to a temporary "
            f"file in your sandbox. Use the read tool to access the full content "
            f"before completing the task. This input file is temporary and will be cleaned up automatically "
            f"after the task completes."
        )

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return (
            f"  - Output `{field_name}`: write your output to a file (e.g. "
            f"/workspace/outputs/{field_name}.txt) and return only "
            f"the file path as the field_name value. This output file is temporary and will be cleaned up automatically "
            f"after the task completes. "
            f"If the output is long, write it in multiple smaller tool calls (e.g. write the first portion, "
            f"then append subsequent portions) rather than generating it all in one response — "
            f"this avoids hitting generation length limits."
        )

    @classmethod
    def get_return_tool_description(cls, field_name: str) -> str:
        return (
            f"Register the '{field_name}' output file. "
            f"Write your output to a sandbox file first (e.g. /workspace/outputs/{field_name}.txt), "
            f"then call this tool with that file path as the value."
        )

    @classmethod
    def get_return_tool_value_description(cls, field_name: str) -> str:
        return (
            f"Absolute path to the file you wrote in the sandbox "
            f"(e.g. /workspace/outputs/{field_name}.txt). Do NOT pass the file content — pass the path."
        )

    @classmethod
    def validate_output(
        cls,
        field_name: str,
        value: object,
        sandbox: "agSandbox",
        exec_timeout: int,
    ) -> "str | None":
        if not isinstance(value, str):
            output = None
        else:
            try:
                content = sandbox.read_file(value)
            except IsADirectoryError:
                output = (
                    f"field_name '{field_name}': '{value}' is a directory, not a file. "
                    f"Pass the path to a specific output file (e.g. {value}/{field_name}.txt)."
                )
            except UnicodeDecodeError:
                output = (
                    f"field_name '{field_name}': file at '{value}' contains binary data "
                    f"and cannot be read as text. Write a UTF-8 text file instead."
                )
            except Exception:
                output = (
                    f"field_name '{field_name}': no file found at path '{value}'. "
                    f"Write your output to a file first, then call this tool with that file's path."
                )
            else:
                if not content or not content.strip():
                    output = (
                        f"field_name '{field_name}': file at '{value}' is empty. "
                        f"Write the actual content to the file before registering the path."
                    )
                elif _looks_like_path(content.strip()):
                    output = (
                        f"field_name '{field_name}': file at '{value}' contains only a path reference "
                        f"('{content.strip()}'), not real content. "
                        f"Write the actual content to a file and return that file's path."
                    )
                else:
                    output = None
        return output


class agpath(agtype):
    """Path-only string field_name for agskill schemas.

    ``agpath`` keeps a path as a path. Unlike ``agfile`` recovery, neither
    prepare() nor recover() reads the sandbox filesystem. The value passes
    through unchanged; only its path-like shape is checked. Plain ``str``
    output also remains text but does not impose this path-shape contract.

    Input fields
    ------------
    The value must look like a path (``_looks_like_path``); anything else
    fails an explicit ``agschema.validate_input()`` check. Skill execution
    does not invoke that check automatically.

    Output fields
    ------------
    The LLM must return a path string, not inline content. If the value
    doesn't look like a path, ``validate_output`` feeds back an error so the
    LLM retries with an actual path.

    Example::

        move_skill = agskill(
            name="move_file",
            prompt="Move the file to the given destination.",
            input_schema=agdata(src=agpath, dest=agpath),
            output_schema=agdata(moved_to=agpath),
        )
    """

    @classmethod
    def schema_type(cls) -> str:
        return "path"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return f"  - Input `{field_name}`: a path string, not file content."

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return (
            f"  - Output `{field_name}`: return the path itself "
            f"(e.g. /workspace/outputs/{field_name}.txt) as the field_name value. "
            f"Do NOT pass file content — pass only the path string."
        )

    @classmethod
    def get_return_tool_description(cls, field_name: str) -> str:
        return f"Register the '{field_name}' output as a path string."

    @classmethod
    def get_return_tool_value_description(cls, field_name: str) -> str:
        return f"Path string for '{field_name}'. Do NOT pass file content — pass only the path."

    @classmethod
    def validate_output(
        cls,
        field_name: str,
        value: object,
        sandbox: "agSandbox",
        exec_timeout: int,
    ) -> "str | None":
        if not isinstance(value, str) or not _looks_like_path(value):
            return (
                f"field_name '{field_name}': '{value}' does not look like a path. "
                f"Pass the path string itself, not file content."
            )
        return None

    @classmethod
    def validate_input_value(cls, value: object) -> "str | None":
        if not isinstance(value, str):
            return f"must be a string, got {type(value).__name__}"
        if not _looks_like_path(value):
            return f"'{value}' does not look like a path"
        return None


class agimage(agtype):
    """Image input field_name for agskill schemas.

    The caller supplies a local file path, an http/https URL, or an existing
    data URL.  Local files are base64-encoded by the framework before the skill
    runs.  The image is injected directly into the multimodal content array of
    the user message — the LLM sees it as a visual input, not as text.

    Single image::

        skill = agskill(
            name="describe",
            prompt="Describe the image.",
            input_schema=agdata(question=str, photo=agimage),
        )

    List of images::

        skill = agskill(
            name="compare",
            prompt="Compare the images.",
            input_schema=agdata(question=str, frames=list[agimage]),
        )
    """

    @classmethod
    def schema_type(cls) -> str:
        return "image"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        # Already a URL or data URL — pass through unchanged.
        if value.startswith(("http://", "https://", "data:")):
            return value, []
        # Local file path — read and base64-encode.
        path = Path(value)
        raw = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        b64 = base64.b64encode(raw).decode()
        return f"data:{mime};base64,{b64}", []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: an image attached directly to this message "
            f"as a visual input. The JSON field_name value is a placeholder — the actual "
            f"image is visible to you in the message content."
        )

    @classmethod
    def build_content_prompt(cls, key: str, value: object) -> "tuple[str | None, list[dict]]":
        if isinstance(value, list):
            urls = [v for v in value if isinstance(v, str)]
            blocks = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
            return f"[{len(blocks)} image(s) attached]", blocks
        if isinstance(value, str):
            return "[image attached]", [{"type": "image_url", "image_url": {"url": value}}]
        return None, []

    @classmethod
    def get_return_tool_description(cls, field_name: str) -> str:
        return f"Register the '{field_name}' image output as a file path or data URL."

    @classmethod
    def get_return_tool_value_description(cls, field_name: str) -> str:
        return f"File path, http/https URL, or data: URL of the image for '{field_name}'."


class agbinary(agtype):
    """Binary file-backed agskill schema field_name.

    Carries raw bytes through the skill boundary via the sandbox filesystem.
    The LLM never sees the binary content — it only sees a file path and is
    told to use shell tools (``file``, ``xxd``, domain-specific CLIs) to
    inspect or produce binary output.

    Caller value types
    ------------------
    Input  (``prepare``): ``bytes``, a local host file path (``str``), or a
    base64 data URL (``str`` starting with ``data:``).

    Output (successful ``recover``): ``bytes``. Schema recovery is best-effort;
    a failed read can leave the original path value and emit a warning.

    Example::

        process_skill = agskill(
            name="process_audio",
            prompt="Trim the audio to the first 10 seconds using ffmpeg.",
            input_schema=agdata(audio=agbinary),
            output_schema=agdata(trimmed=agbinary),
        )

        result = ag.run(process_skill, agdata(audio=Path("clip.wav").read_bytes()))
        Path("trimmed.wav").write_bytes(result.trimmed)

    Note: the agent's sandbox image must include whatever CLI tools are needed
    to process the binary format (e.g. ``ffmpeg``, ``imagemagick``, ``sox``).
    """

    @classmethod
    def schema_type(cls) -> str:
        return "binary_file"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return True

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
        suffix: str = "",
    ) -> tuple[str, list[str]]:
        path = f"/workspace/inputs/{field_name}{suffix}.bin"
        try:
            raw = cls._to_bytes(value)
        except (TypeError, ValueError):
            return value, []
        try:
            sandbox.write_file_bytes(path, raw)
            return path, [path]
        except Exception as _e:
            print(f"[agtype] WARNING: agbinary.prepare failed to write {path}: {_e}")
            return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[bytes, list[str]]:
        if not isinstance(value, str):
            return value, []
        try:
            raw = sandbox.read_file_bytes(value)
            return raw, [value]
        except Exception as _e:
            print(f"[agtype] WARNING: agbinary.recover failed to read {value}: {_e}")
            return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: a binary file at the path shown in the JSON. "
            f"Do NOT read it as text — use shell tools (e.g. `file`, `xxd`, or "
            f"domain-specific CLIs) to inspect or process it. "
            f"This file is temporary and will be deleted after the task completes."
        )

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return (
            f"  - Output `{field_name}`: write your binary output to a file "
            f"(e.g. /workspace/outputs/{field_name}.bin) using shell tools, "
            f"then return only the file path as the field_name value. "
            f"Do NOT encode the content as text or base64 — write the raw binary file."
        )

    @classmethod
    def get_return_tool_description(cls, field_name: str) -> str:
        return (
            f"Register the '{field_name}' binary output file. "
            f"Write your output as a binary file first "
            f"(e.g. /workspace/outputs/{field_name}.bin), "
            f"then call this tool with that file path as the value."
        )

    @classmethod
    def get_return_tool_value_description(cls, field_name: str) -> str:
        return (
            f"Absolute path to the binary file you wrote in the sandbox "
            f"(e.g. /workspace/outputs/{field_name}.bin). "
            f"Pass the path — do NOT pass encoded content."
        )

    @classmethod
    def validate_output(
        cls,
        field_name: str,
        value: object,
        sandbox: "agSandbox",
        exec_timeout: int,
    ) -> "str | None":
        if not isinstance(value, str):
            return None
        _, dir_rc = sandbox._container_exec(
            f"test -d {shlex.quote(value)}", timeout=exec_timeout, shell="sh"
        )
        if dir_rc == 0:
            return (
                f"field_name '{field_name}': '{value}' is a directory, not a file. "
                f"Pass the path to a specific binary output file (e.g. {value}/{field_name}.bin)."
            )
        _, exist_rc = sandbox._container_exec(
            f"test -s {shlex.quote(value)}", timeout=exec_timeout, shell="sh"
        )
        if exist_rc != 0:
            _, found_rc = sandbox._container_exec(
                f"test -e {shlex.quote(value)}", timeout=exec_timeout, shell="sh"
            )
            if found_rc != 0:
                return (
                    f"field_name '{field_name}': no file found at path '{value}'. "
                    f"Write your binary output to a file first, then call this tool with that file's path."
                )
            return (
                f"field_name '{field_name}': file at '{value}' is empty. "
                f"Write the actual binary content to the file before registering the path."
            )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _to_bytes(cls, value: object) -> bytes:
        """Normalise caller input to raw bytes."""
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            if value.startswith("data:") and ";base64," in value:
                _, b64 = value.split(";base64,", 1)
                return base64.b64decode(b64)
            # Treat as a local host file path.
            return Path(value).read_bytes()
        raise TypeError(f"agbinary.prepare: unsupported value type {type(value).__name__!r}")


class agrawstring(agtype):
    """Raw string field_name — bypasses JSON input/output formatting entirely.

    Input
    -----
    When used as the sole input field_name, the string value is sent directly as
    the user message content.  No JSON wrapping, no schema type_hint.

    Output
    ------
    When used as the sole output field_name, the model's complete text response is
    captured as-is.  No JSON parsing, no retry loop.

    Constraint
    ----------
    Must be the only field_name in its input or output schema.

    Example::

        write_skill = agskill(
            name="write_chapter",
            prompt="You are a novelist. Write the chapter as requested.",
            input_schema=agdata(prompt=agrawstring),
            output_schema=agdata(chapter=agrawstring),
        )

        result = ag.run(write_skill, agdata(prompt="Write a dark opening scene."))
        print(result.chapter)   # the model's prose, unmodified
    """

    @classmethod
    def schema_type(cls) -> str:
        return "str"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False


# ---------------------------------------------------------------------------
# Type-type_hint inspection and schema helpers
# ---------------------------------------------------------------------------


def type_hint_to_string_type(type_hint) -> str:
    """Map a schema type_hint to a JSON Schema type string for tool parameter specs."""
    if isinstance(type_hint, type):
        if issubclass(type_hint, bool):
            output = "boolean"  # bool before int (bool is subclass of int)
        elif issubclass(type_hint, int):
            output = "integer"
        elif issubclass(type_hint, float):
            output = "number"
        elif issubclass(type_hint, agtype):
            output = "string"
        elif issubclass(type_hint, (list, tuple)):
            output = "array"
        elif issubclass(type_hint, dict):
            output = "object"
        else:
            output = "string"
    else:
        origin = get_origin(type_hint)
        if origin is list or origin is tuple:
            output = "array"
        elif origin is dict:
            output = "object"
        elif isinstance(type_hint, list):
            output = "array"  # [{"key": type, ...}] literal
        else:
            output = "string"
    return output


def get_json_example_for_type_hint(type_hint) -> str:
    """Return a short valid-JSON example for a type type_hint (used in tool descriptions/errors).

    Every returned string is parseable by json.loads.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if type_hint is bool:
        output = "true"
    elif type_hint is int:
        output = "42"
    elif type_hint is float:
        output = "3.14"
    elif type_hint is str:
        output = '"The capital of France is Paris."'
    elif type_hint is list or origin is list:
        output = f"[{get_json_example_for_type_hint(args[0])}]" if args else "[]"
    elif type_hint is dict or origin is dict:
        if args and len(args) == 2:
            output = (
                "{"
                + f"{get_json_example_for_type_hint(args[0])}: {get_json_example_for_type_hint(args[1])}"
                + "}"
            )
        else:
            output = "{}"
    elif type_hint is tuple or origin is tuple:
        output = (
            "[" + ", ".join(get_json_example_for_type_hint(t) for t in args) + "]" if args else "[]"
        )
    elif isinstance(type_hint, list) and len(type_hint) == 1 and isinstance(type_hint[0], dict):
        obj = (
            "{"
            + ", ".join(
                f'"{k}": {get_json_example_for_type_hint(t)}' for k, t in type_hint[0].items()
            )
            + "}"
        )
        output = f"[{obj}]"
    elif isinstance(type_hint, type) and issubclass(type_hint, agtype):
        output = '"value"'
    else:
        output = "null"
    return output


def get_value_format_prompt_for_type_hint(field_name: str, type_hint) -> str:
    """Return a value description with a concrete format example and a 'pass directly' note."""
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    ex = get_json_example_for_type_hint(type_hint)
    direct = "Pass directly — do not JSON-encode into a string."

    if type_hint is str:
        output = (
            f"The complete string value for '{field_name}'. "
            "Pass the full content directly — not a file path."
        )
    elif type_hint is bool:
        output = f"Boolean for '{field_name}' (true or false)."
    elif type_hint is int:
        output = f"Integer for '{field_name}'. Example: {ex}."
    elif type_hint is float:
        output = f"Floating-point number for '{field_name}'. Example: {ex}."
    elif type_hint is list or origin is list:
        if args:
            elem_type = args[0]
            elem_name = getattr(elem_type, "__name__", repr(elem_type))
            output = f"JSON array of {elem_name} values for '{field_name}'. Example: {ex}. {direct}"
        else:
            output = f"JSON array for '{field_name}'. Example: {ex}. {direct}"
    elif type_hint is dict or origin is dict:
        if args and len(args) == 2:
            kn = getattr(args[0], "__name__", repr(args[0]))
            vn = getattr(args[1], "__name__", repr(args[1]))
            output = (
                f"JSON object with {kn} keys and {vn} values for '{field_name}'. "
                f"Example: {ex}. {direct}"
            )
        else:
            output = f"JSON object for '{field_name}'. Example: {ex}. {direct}"
    elif type_hint is tuple or origin is tuple:
        if args:
            types_str = ", ".join(getattr(t, "__name__", repr(t)) for t in args)
            output = (
                f"JSON array of {len(args)} element(s) ({types_str}) for '{field_name}'. "
                f"Example: {ex}. {direct}"
            )
        else:
            output = f"JSON array for '{field_name}'. Example: {ex}. {direct}"
    elif isinstance(type_hint, list) and len(type_hint) == 1 and isinstance(type_hint[0], dict):
        keys = ", ".join(f'"{k}"' for k in type_hint[0])
        obj_ex = (
            "{"
            + ", ".join(
                f'"{k}": {get_json_example_for_type_hint(t)}' for k, t in type_hint[0].items()
            )
            + "}"
        )
        output = (
            f"JSON array of objects for '{field_name}'. Each object must have keys: {keys}. "
            f"Example: [{obj_ex}]. {direct}"
        )
    else:
        output = f"Value for '{field_name}'."
    return output


def get_return_tool_description_prompt(field_name: str, type_hint) -> "tuple[str, str]":
    """Return (tool_description, value_description) for a return_<field_name> tool."""
    ex = get_json_example_for_type_hint(type_hint)
    tool_desc = (
        f"Use this tool to return the final value for the '{field_name}' output field, by supplying it as an argument to this tool. "
        f"You must pass the actual output content as the '{field_name}' argument. Simply outputting the context and calling this tool without passing an argument will not work. "
        f'Format Example (JSON): {{"{field_name}": {ex}}}'
    )
    # agtype subclass — delegate to its classmethods
    if isinstance(type_hint, type) and issubclass(type_hint, agtype):
        return type_hint.get_return_tool_description(
            field_name
        ), type_hint.get_return_tool_value_description(field_name)
    # list[agtype] — delegate to the inner type, but include a JSON array example
    if get_origin(type_hint) is list:
        args = get_args(type_hint)
        if args and isinstance(args[0], type) and issubclass(args[0], agtype):
            inner = args[0]
            ex = get_json_example_for_type_hint(type_hint)
            return (
                inner.get_return_tool_description(field_name) + " (as a JSON array)",
                inner.get_return_tool_value_description(field_name)
                + f" Provide as a JSON array. Example: {ex}.",
            )
    return tool_desc, get_value_format_prompt_for_type_hint(field_name, type_hint)


def validate_value_against_type_hint(type_hint, value) -> "str | None":
    """Recursively validate value against type_hint. Returns an error string or None."""
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if isinstance(type_hint, type):
        if issubclass(type_hint, agtype):
            output = type_hint.validate_input_value(value)
        elif issubclass(type_hint, bool):
            output = (
                None if isinstance(value, bool) else f"expected bool, got {type(value).__name__}"
            )
        elif issubclass(type_hint, (int, float, str)):
            if isinstance(value, type_hint):
                output = None
            elif issubclass(type_hint, float) and isinstance(value, int):
                output = None  # int is accepted for float fields; caller receives the int, which is a valid float
            else:
                output = f"expected {type_hint.__name__}, got {type(value).__name__}"
        # bare list/tuple/dict without type args
        elif issubclass(type_hint, (list, tuple)):
            output = (
                None
                if isinstance(value, (list, tuple))
                else f"expected array, got {type(value).__name__}"
            )
        elif issubclass(type_hint, dict):
            output = (
                None if isinstance(value, dict) else f"expected dict, got {type(value).__name__}"
            )
        else:
            output = (
                None
                if isinstance(value, type_hint)
                else f"expected {type_hint.__name__}, got {type(value).__name__}"
            )
    elif origin is list:
        if not isinstance(value, list):
            output = f"expected list, got {type(value).__name__}"
        else:
            output = None
            if args:
                for i, item in enumerate(value):
                    err = validate_value_against_type_hint(args[0], item)
                    if err:
                        output = f"item {i}: {err}"
                        break
    elif origin is tuple:
        if not isinstance(value, (list, tuple)):
            output = f"expected array, got {type(value).__name__}"
        else:
            output = None
            if args:
                for i, (type_arg, item) in enumerate(zip(args, value)):
                    err = validate_value_against_type_hint(type_arg, item)
                    if err:
                        output = f"item {i}: {err}"
                        break
    elif origin is dict:
        if not isinstance(value, dict):
            output = f"expected dict, got {type(value).__name__}"
        else:
            output = None
            if len(args) == 2:
                for k, v in value.items():
                    err = validate_value_against_type_hint(args[1], v)
                    if err:
                        output = f"key {k!r}: {err}"
                        break
    # [{"key": type, ...}] literal list-of-dicts schema
    elif isinstance(type_hint, list) and len(type_hint) == 1 and isinstance(type_hint[0], dict):
        if not isinstance(value, list):
            output = f"expected list, got {type(value).__name__}"
        else:
            output = None
            template = type_hint[0]
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    output = f"item {i}: expected dict, got {type(item).__name__}"
                else:
                    for k, t in template.items():
                        if k not in item:
                            output = f"item {i}: missing key '{k}'"
                        elif isinstance(t, type) and not isinstance(item[k], t):
                            output = (
                                f"item {i}.{k}: expected {t.__name__}, got {type(item[k]).__name__}"
                            )
                        if output is not None:
                            break
                if output is not None:
                    break
    else:
        output = None

    return output


def validate_output_field_against_schema(field_name: str, value, schema) -> "str | None":
    """Validate a single output field value against its type hint in schema._data."""
    return validate_value_against_type_hint(schema._data[field_name], value)


# ---------------------------------------------------------------------------
# Type-type_hint traversal helpers  (now static methods on agtype)
# ---------------------------------------------------------------------------


def output_field_desc(type_hint: object) -> str:
    """Return a human-readable type description with usage guidance for an output field_name."""
    if isinstance(type_hint, type) and issubclass(type_hint, agtype):
        output = type_hint.schema_type()
    elif type_hint is str:
        output = "string — pass the complete text content as the value (not a file path)"
    elif type_hint is int:
        output = "integer — pass the numeric value directly"
    elif type_hint is float:
        output = "float — pass the numeric value directly"
    elif type_hint is bool:
        output = "boolean — pass true or false"
    elif type_hint is list or type_hint is dict:
        output = type_hint.__name__
    elif get_origin(type_hint) is list:
        args = get_args(type_hint)
        inner = output_field_desc(args[0]) if args else "any"
        output = f"array of {inner}"
    else:
        output = str(type_hint)
    return output
