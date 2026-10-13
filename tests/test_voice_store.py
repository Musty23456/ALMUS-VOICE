import pytest

from app.services.voice_store import (
    InvalidReferenceAudioError,
    InvalidVoiceIdError,
    VoiceExistsError,
    VoiceNotFoundError,
    VoiceStore,
)
from tests.helpers import make_wav


def test_save_and_read_back(store: VoiceStore) -> None:
    info = store.save("almus_owner", make_wav(6))
    assert info.sample_rate == 16000 and info.channels == 1
    assert store.get_path("almus_owner").is_file()
    assert store.list_ids() == ["almus_owner"]


@pytest.mark.parametrize(
    "bad", ["../evil", "a/b", "a\\b", "..", "a.b", "", "x" * 65, "sp ace", "voice\x00"]
)
def test_unsafe_voice_ids_rejected(store: VoiceStore, bad: str) -> None:
    with pytest.raises(InvalidVoiceIdError):
        store.path_for(bad)


def test_files_stay_inside_root(store: VoiceStore, tmp_path) -> None:
    store.save("ok-id", make_wav(6))
    assert store.get_path("ok-id").parent == (tmp_path / "voices").resolve()


def test_rejects_non_wav(store: VoiceStore) -> None:
    with pytest.raises(InvalidReferenceAudioError):
        store.save("v", b"ID3 not a wav at all")


def test_rejects_too_short_and_too_long(store: VoiceStore) -> None:
    with pytest.raises(InvalidReferenceAudioError, match="too short"):
        store.save("v", make_wav(1))
    with pytest.raises(InvalidReferenceAudioError, match="too long"):
        store.save("v", make_wav(61, rate=8000 * 2))


def test_rejects_low_sample_rate(store: VoiceStore) -> None:
    with pytest.raises(InvalidReferenceAudioError, match="Sample rate"):
        store.save("v", make_wav(6, rate=8000))


def test_rejects_oversize(store: VoiceStore) -> None:
    with pytest.raises(InvalidReferenceAudioError, match="too large"):
        store.save("v", b"\0" * (store.max_bytes + 1))


def test_overwrite_requires_flag(store: VoiceStore) -> None:
    store.save("v", make_wav(6))
    with pytest.raises(VoiceExistsError):
        store.save("v", make_wav(6))
    store.save("v", make_wav(7), overwrite=True)


def test_delete_and_missing(store: VoiceStore) -> None:
    store.save("v", make_wav(6))
    store.delete("v")
    with pytest.raises(VoiceNotFoundError):
        store.get_path("v")


def test_no_temp_files_left(store: VoiceStore, tmp_path) -> None:
    store.save("v", make_wav(6))
    names = [p.name for p in (tmp_path / "voices").iterdir()]
    assert names == ["v.wav"]
