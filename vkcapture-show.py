import socket
import struct
import mmap
import os
import fcntl
import numpy as np
import cv2
import select
import gc

# --- Constants ---
SOCKET_PATH = '\0/com/obsproject/vkcapture'
CTRL_FMT = '<BBBB16s12x'
TEX_FMT = '<BBiii4i4iQIBI65x'
TEX_SIZE = 128
TYPE_TEXTURE_DATA = 11

DMA_BUF_IOCTL_SYNC = 0x40086200
DMA_BUF_SYNC_READ = 1
DMA_BUF_SYNC_START = 0
DMA_BUF_SYNC_END = 4

def dma_sync(fd, flags):
    try:
        sync_args = struct.pack('Q', flags)
        fcntl.ioctl(fd, DMA_BUF_IOCTL_SYNC, sync_args)
    except OSError:
        pass

def main():
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setblocking(False)
    try:
        server.bind(SOCKET_PATH)
    except OSError as e:
        print(f"Error binding socket: {e}")
        return
    server.listen(1)
    print("Viewer started. Waiting for game...")

    # State variables
    conn = None
    mapped_buf = None
    current_fd = None
    width, height, stride = 0, 0, 0

    # We must track these views to delete them before closing the buffer
    raw = None
    img = None

    try:
        while True:
            readers = [server]
            if conn:
                readers.append(conn)

            readable, _, _ = select.select(readers, [], [], 0.01)

            for s in readable:
                if s is server:
                    # --- NEW CONNECTION (RESTART) ---
                    new_conn, _ = server.accept()
                    print("Game connected!")

                    # Clean up previous connection completely
                    if conn:
                        conn.close()

                        # 1. Release Views
                        raw = None
                        img = None
                        del raw
                        del img
                        gc.collect()

                        # 2. Close Map and FD
                        if mapped_buf:
                            try: mapped_buf.close()
                            except ValueError: pass

                        if current_fd:
                            os.close(current_fd)

                        # 3. CRITICAL FIX: Reset state variables to None
                        mapped_buf = None
                        current_fd = None
                        width, height = 0, 0

                    conn = new_conn
                    conn.setblocking(True)

                    try:
                        msg = struct.pack(CTRL_FMT, 1, 0, 1, 1, b'\0'*16)
                        conn.send(msg)
                    except BrokenPipeError:
                        conn = None
                        print("Handshake failed.")

                elif s is conn:
                    # --- DATA FROM GAME ---
                    try:
                        data, ancdata, flags, addr = conn.recvmsg(TEX_SIZE, socket.CMSG_LEN(struct.calcsize('i') * 4))

                        if not data:
                            print("Game disconnected.")
                            conn.close()
                            conn = None

                            # Clean up
                            raw = None
                            img = None
                            del raw
                            del img
                            gc.collect()

                            if mapped_buf:
                                try: mapped_buf.close()
                                except ValueError: pass

                            if current_fd:
                                os.close(current_fd)

                            mapped_buf = None
                            current_fd = None
                            width, height = 0, 0
                            continue

                        if data[0] == TYPE_TEXTURE_DATA and len(data) == TEX_SIZE:
                            fields = struct.unpack(TEX_FMT, data)
                            new_w, new_h, new_stride = fields[2], fields[3], fields[5]

                            fds = []
                            for cmsg_level, cmsg_type, cmsg_data in ancdata:
                                if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                                    fds.extend(struct.unpack('i' * (len(cmsg_data) // 4), cmsg_data))

                            if fds:
                                # --- RESIZE EVENT ---
                                if current_fd: os.close(current_fd)

                                raw = None
                                img = None
                                del raw
                                del img
                                gc.collect()

                                if mapped_buf:
                                    try: mapped_buf.close()
                                    except ValueError: pass
                                mapped_buf = None # Reset strictly

                                current_fd = fds[0]
                                size = os.lseek(current_fd, 0, os.SEEK_END)
                                os.lseek(current_fd, 0, os.SEEK_SET)
                                mapped_buf = mmap.mmap(current_fd, size, mmap.MAP_SHARED, mmap.PROT_READ)

                                width, height, stride = new_w, new_h, new_stride
                                print(f"Texture update: {width}x{height} (FD: {current_fd})")

                    except (ConnectionResetError, BrokenPipeError):
                        print("Connection lost.")
                        conn.close()
                        conn = None
                        mapped_buf = None

            # --- RENDER FRAME ---
            # Now safe because mapped_buf is None if we are between games
            if conn and mapped_buf and width > 0 and height > 0:
                try:
                    dma_sync(current_fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)

                    raw = np.frombuffer(mapped_buf, dtype=np.uint8)
                    expected_size = stride * height

                    if raw.size >= expected_size:
                        img = raw[:expected_size].reshape((height, stride))
                        img = img[:, :width*4]
                        img = img.reshape((height, width, 4))
                        img = img[..., [2, 1, 0, 3]]

                        cv2.imshow("OBS VkCapture Output", img)

                    dma_sync(current_fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ)

                except ValueError:
                    # Handle resize race conditions
                    pass
                except OSError:
                    pass

            # GUI Loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Final Cleanup
        raw = None
        img = None
        del raw
        del img
        gc.collect()
        if mapped_buf:
            try: mapped_buf.close()
            except ValueError: pass
        if current_fd: os.close(current_fd)
        if conn: conn.close()
        server.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
