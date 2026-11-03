import json
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from robofape.artifacts import digest,resource,write_json,validate_trajectory
from robofape.cli import label_command,main
from robofape.config import task_ids,load_config
from robofape.expressions import SafeExpression,UnsafeExpressionError
from robofape.policy import decode_array,encode_array,receive,send,serve,valid_actions,MAX_MESSAGE


class InterfaceTest(unittest.TestCase):
    def test_all_twenty_configs_are_packaged_and_valid(self):
        self.assertEqual(len(task_ids()),20)
        for task in task_ids():
            c=load_config(task);self.assertEqual(c["num_stage"],len(c["stage_order"]))
            self.assertEqual(len(c["sha256"]),64)

    def test_expression_rejects_execution(self):
        for expr in ["__import__('os')","x.__class__","open(x)","x[0]"]:
            with self.assertRaises(UnsafeExpressionError):SafeExpression(expr,{"x"})

    def test_move_dataset_and_label_from_external_workdir(self):
        with tempfile.TemporaryDirectory(prefix="space path ") as tmp:
            root=Path(tmp);src=root/"data";src.mkdir()
            ref={"id":"reference","task_id":"example","success":True,"stage":["a"]*4,"distance":[3,2,1,0]}
            fail={"id":"failure","task_id":"example","success":False,"stage":["a"]*3,"distance":[2,0,3]}
            # Generic sidecars exercise relocation without claiming real simulator data.
            (src/"sidecars").mkdir()
            (src/"sidecars/frame.bin").write_bytes(b"path portability fixture")
            fail["sha256"]={"sidecars/frame.bin":digest(src/"sidecars/frame.bin")}
            write_json(src/"ref.json",ref);write_json(src/"fail.json",fail)
            write_json(src/"bank.json",{"trajectories":["ref.json"]})
            moved=root/"moved dataset";shutil.move(str(src),moved)
            work=root/"unrelated workdir";work.mkdir()
            result=subprocess.run([sys.executable,"-m","robofape","label",
                "--input","../moved dataset/fail.json","--references","../moved dataset/bank.json",
                "--output","../out"],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            validate_trajectory(root/"out/failure/trajectory.json")
            out=json.loads((root/"out/failure/trajectory.json").read_text())
            self.assertEqual(out["progress"],[.5,1,.25])
            with self.assertRaises(FileExistsError):label_command(moved/"fail.json",moved/"bank.json",root/"out")
            shutil.rmtree(moved)
            relocated=root/"relocated output";shutil.move(str(root/"out"),relocated)
            result=subprocess.run([sys.executable,"-m","robofape","validate",
                "../relocated output/manifest.json"],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((relocated/"failure/sidecars/frame.bin").read_bytes(),b"path portability fixture")

    def test_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"file").write_bytes(b"fixture")
            (root/"nested").mkdir()
            for p in ["/private/file","../file","nested/../file","", ".", "C:/file", "C:file", r"C:\file",r"..\file",r"\\server\share\file"]:
                with self.subTest(path=p),self.assertRaises(ValueError):resource(root,p)
            (root/"nested/escape").symlink_to(root/"file")
            with self.assertRaises(ValueError):resource(root/"nested","escape")

    def test_checkpoint_members_must_be_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weights=root/"weights.bin";weights.write_bytes(b"fixture")
            config={"control_mode":"pd_joint_pos","action_dim":8,"state_dim":9,
                    "task_ids":["PushCube-v1"],"execute_chunk":1}
            with patch("robofape.policy.load_factory") as factory:
                for member in [str(weights),"sub/../weights.bin","C:/weights.bin",r"sub\weights.bin"]:
                    with self.subTest(member=member),self.assertRaises(ValueError):
                        serve(root/"policy.sock",root,{**config,"checkpoint_files":[member]},"unused:factory")
                factory.assert_not_called()
                factory.side_effect=RuntimeError("valid relative path reached factory")
                with self.assertRaisesRegex(RuntimeError,"valid relative path reached factory"):
                    serve(root/"policy.sock",root,{**config,"checkpoint_files":["weights.bin"]},"unused:factory")

    def test_array_serialization_and_validation(self):
        value=np.arange(12,dtype=np.float32).reshape(3,4)
        np.testing.assert_equal(decode_array(encode_array(value)),value)
        malformed=encode_array(value);malformed["shape"]=[1]
        with self.assertRaises(ValueError):decode_array(malformed)
        with self.assertRaises(ValueError):valid_actions([[float("nan")]*8],8)
        with self.assertRaises(ValueError):valid_actions([[0]*7],8)

    def test_fragmented_header_and_eof(self):
        left,right=socket.socketpair()
        raw=json.dumps({"hello":1}).encode();frame=struct.pack("!I",len(raw))+raw
        def writer():
            with left:
                for byte in frame:left.sendall(bytes([byte]))
        thread=threading.Thread(target=writer);thread.start()
        with right:self.assertEqual(receive(right),{"hello":1})
        thread.join()
        a,b=socket.socketpair();a.sendall(b'\x00\x00');a.close()
        with b:
            with self.assertRaises(ConnectionError):receive(b)

    def test_oversized_header(self):
        a,b=socket.socketpair();a.sendall(struct.pack("!I",MAX_MESSAGE+1));a.close()
        with b:
            with self.assertRaises(ValueError):receive(b)

    def test_cli_errors_are_nonzero(self):
        self.assertEqual(main(["validate","not-a-real-file.json"]),1)


if __name__=="__main__":unittest.main()
