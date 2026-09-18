"""Tests for voice_pipeline.adapters.respeaker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import usb.util

from voice_pipeline.adapters import respeaker

_EXPECTED_REQUEST_TYPE = usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE


def test_reset_sends_reboot_control_transfer():
    dev = MagicMock()
    # 첫 find 는 장치 반환, 이후(사라짐 폴링)는 None
    with patch("usb.core.find", side_effect=[dev, None]) as find:
        assert respeaker.reset() is True

    find.assert_called_with(idVendor=0x2886)
    dev.ctrl_transfer.assert_called_once_with(_EXPECTED_REQUEST_TYPE, 0, 7, 48, [1], 1000)


def test_reset_returns_false_when_device_missing():
    with patch("usb.core.find", return_value=None):
        assert respeaker.reset() is False


def test_reset_returns_false_on_permission_error():
    dev = MagicMock()
    dev.ctrl_transfer.side_effect = PermissionError("Access denied")
    with patch("usb.core.find", return_value=dev):
        assert respeaker.reset() is False


def test_wait_present_returns_true_once_device_appears():
    dev = MagicMock()
    with patch("usb.core.find", side_effect=[None, None, dev]):
        assert respeaker.wait_present(timeout_sec=2.0) is True


def test_wait_present_times_out():
    with patch("usb.core.find", return_value=None):
        assert respeaker.wait_present(timeout_sec=0.1) is False
