import os
import sys
import subprocess
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=str, default="rr1")
parser.add_argument('--vc', type=str, default="resrchvc")
parser.add_argument('--user', type=str, default="v-bonli") ## replace to your alias
parser.add_argument('--passwd', type=str, default='87341918@Lsy') ## enter your password here
parser.add_argument('--jobname', type=str, default='fangcha_ddi10')
parser.add_argument('--script', type=str, default='fangcha_ddi.sh')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--extra-params', default="")
args = parser.parse_args()


gpu_per_node = 4
if args.cluster == "rr1" or args.cluster == "eu3" or args.cluster == "resrchprojvc7":
    gpu_per_node = 8

os.environ["PHILLY_VC"] = args.vc
remote_folder = r"/blob2/v-bonli/scripts"  ## specify you own folder

remote_script = r'{0}/{1}'.format(remote_folder, args.script)

job_json={
    "ClusterId": "{}".format(args.cluster),
    "VcId": "{}".format(args.vc),
    "JobName": "{}".format(args.jobname),
    "UserName": args.user,
    "noProgressTimeout": 2592000,
    "BuildId": 0,
    "ToolType": None,
    "ConfigFile": "{}".format(remote_script),
    "Inputs": [{
        "Name": "dataDir",
        "Path": remote_folder
        }],
    "Outputs": [],
    "IsDebug": args.debug,
    "RackId": "anyConnected",
    "MinGPUs": args.gpu,
    "PrevModelPath": None,
    "ExtraParams": args.extra_params,
    "SubmitCode": "p",
    "IsMemCheck": False,
    "IsCrossRack": False,
    "Registry": "phillyregistry.azurecr.io",
    "Repository": "philly/jobs/custom/pytorch",
    "Tag": "pytorch1.5-py36-cuda10.1-apex",
    "OneProcessPerContainer": True,
    "DynamicContainerSize": False,
    "NumOfContainers": max(args.gpu // gpu_per_node, 1),
    "CustomMPIArgs": None,
    "Timeout": None,
    "volumes": {
		"myblob1": {
            "_comment": "This will mount testcontainer in the storage account pavermatest inside the container at path '/blob2'. The credentials required for accessing storage account pavermatest are below, in the 'credentials' section.",
            "type": "blobfuseVolume",
            "storageAccount": "msralaphilly2",
            "containerName": "ml-la",
            "path": "/blob2"
        },
        "myblob2": {
            "_comment": "This will mount testcontainer in the storage account pavermatest inside the container at path '/blob3'. The credentials required for accessing storage account pavermatest are below, in the 'credentials' section.",
            "type": "blobfuseVolume",
            "storageAccount": "msralascv2",
            "containerName": "ml-la",
            "path": "/blob3"
        },
    },
    "credentials": {
        "storageAccounts": {
            "msralaphilly2": {
                "_comment": "Credentials for accessing 'pavermatest' storage account. Secrets can be saved with Philly from your Philly profile page at https://philly/#/userView/. With this the secret doesn't have to be maintained in the user's workspace.",
                "key": "iFMeu8F2oorziqr5eD1igS3GVggk6TJPdH0N4pPggoJZolXFKOg55NdsjieS1T7yq4y5YddHHfCbDJypGgYIoA=="
            },
            "msralascv2": {
                "_comment": "Credentials for accessing 'pavermatest' storage account. Secrets can be saved with Philly from your Philly profile page at https://philly/#/userView/. With this the secret doesn't have to be maintained in the user's workspace.",
                "key": "ruxZ8peIL7ntvFiakowgxB2dQ2MRckMkpT2FBnP+zmbE5LAh8v4WFoQ4ggoRbafVWRnwF2degK4zOpyN5O5HUA=="
            }
        }
    }
}

with open('params.json', 'w') as f:
    json.dump(job_json, f)

os.system('curl -k --ntlm --user {0}:{1} -X POST -H "Content-Type: application/json" --data @params.json https://philly/api/v2/submit'.format(args.user, args.passwd))
