"""
SIYI A8mini 控制SDK
协议参考: SDK - V0.1.1.pdf
"""

import socket
import struct
import time
import logging

log = logging.getLogger("tracker.siyi")


class SIYIA8mini:
    def __init__(self, ip="192.168.144.25", port=37260):
        self.ip = ip
        self.port = port
        self.rtsp_url = f"rtsp://{ip}:8554/main.264"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self.seq = 0

    def _crc16(self, data):
        """CRC16 校验 (与SDK实例匹配)"""
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    def _build_packet(self, cmd_id, data=b''):
        """
        构建SIYI协议包
        格式: STX(2) + CTRL(1) + Data_len(2 LE) + SEQ(2 LE) + CMD_ID(1) + DATA + CRC16(2 LE)
        """
        self.seq = (self.seq + 1) & 0xFFFF
        data_len = len(data)  # 只算DATA长度，不含CMD_ID
        # STX=0x6655 低字节在前 -> 55 66
        packet = bytearray()
        packet += b'\x55\x66'                           # STX
        packet += struct.pack('B', 0x01)                # CTRL: need_ack=1
        packet += struct.pack('<H', data_len)           # Data_len (LE)
        packet += struct.pack('<H', self.seq)           # SEQ (LE)
        packet += struct.pack('B', cmd_id)              # CMD_ID
        packet += data                                  # DATA
        crc = self._crc16(packet)
        packet += struct.pack('<H', crc)                # CRC16 (LE)
        return bytes(packet)

    def _send(self, cmd_id, data=b''):
        """发送指令并等待应答"""
        packet = self._build_packet(cmd_id, data)
        self.sock.sendto(packet, (self.ip, self.port))
        try:
            resp, _ = self.sock.recvfrom(1024)
            return resp
        except socket.timeout:
            return None

    def zoom_in(self):
        """手动变焦放大 (CMD 0x05, data=1)"""
        # SDK实例: 55 66 01 01 00 00 00 05 01 8d 64
        resp = self._send(0x05, struct.pack('b', 1))
        return resp

    def zoom_out(self):
        """手动变焦缩小 (CMD 0x05, data=-1)"""
        # SDK实例: 55 66 01 01 00 00 00 05 FF 5c 6a
        resp = self._send(0x05, struct.pack('b', -1))
        return resp

    def zoom_stop(self):
        """停止变焦 (CMD 0x05, data=0)"""
        resp = self._send(0x05, struct.pack('b', 0))
        return resp

    def set_zoom(self, zoom_level):
        """
        绝对变焦 (CMD 0x0F)
        zoom_level: float, 如 1.0, 2.0, 4.5, 6.0
        data: uint8 整数部分 + uint8 小数部分
        SDK实例: 4.5x -> 55 66 01 02 00 01 00 0F 04 05 60 BB
        """
        int_part = int(zoom_level)
        float_part = int(round((zoom_level - int_part) * 10))
        data = struct.pack('BB', int_part, float_part)
        resp = self._send(0x0F, data)
        if resp:
            log.info("变焦: %.1fx (已应答)", zoom_level)
        else:
            log.warning("变焦: %.1fx (无应答)", zoom_level)
        return resp

    def get_zoom(self):
        """获取当前变焦倍数 (CMD 0x18)"""
        resp = self._send(0x18)
        if resp and len(resp) >= 10:
            # 打印原始应答帮助调试
            hex_str = ' '.join(f'{b:02x}' for b in resp)
            log.debug("get_zoom 原始应答: %s", hex_str)
            # ACK格式: STX(2)+CTRL(1)+DataLen(2)+SEQ(2)+CMD_ID(1)+DATA+CRC(2)
            # DATA位于第8字节开始: zoom_int(1) + zoom_float(1)
            try:
                zoom_int = resp[8]
                zoom_float = resp[9]
                return zoom_int + zoom_float / 10.0
            except IndexError:
                pass
        return None

    def close(self):
        self.sock.close()
