# Text Classifier — CIL ETH Zürich

> **Getting started guide** — cluster setup, environment configuration, and notebook workflow.

---

## Cluster Paths

| Resource | Path |
|---|---|
| Conda environment | `/cluster/courses/cil/envs/envs/text-classification` |
| Data | `/cluster/courses/cil/text-classification/data/` |

## Useful Links

| Resource | URL |
|---|---|
| Cluster documentation | https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster |
| Project Google Doc | https://docs.google.com/document/d/1JL8J60MP6A4DblW66prtFsrrRMduYGAZdayUqy5QahA/edit?tab=t.0 |
| Kaggle competition | https://www.kaggle.com/t/7aafec306ca648acacee5c4fe633c0fb |
| Jupyter Hub (GPU) | https://student-jupyter.inf.ethz.ch/hub/spawn |

---

## 1 — Connect to the Cluster

You can connect directly with:

_Run on your local machine:_
```bash
$ ssh <eth-username>@student-cluster.inf.ethz.ch
```
> Replace `<eth-username>` with your ETH username (e.g. `jdoe`).

You will be prompted for your ETH password each time. To avoid this, follow the steps below to set up SSH key authentication.

---

### 1.1 — Check for an existing SSH key

_Run on your local machine:_
```bash
$ ls ~/.ssh/id_ed25519.pub
```

- If the file **exists** → skip to [1.3](#13--copy-your-key-to-the-cluster).
- If you get `No such file or directory` → continue to [1.2](#12--generate-an-ssh-key).

---

### 1.2 — Generate an SSH key

_Run on your local machine:_
```bash
$ ssh-keygen -t ed25519 -C "key-ethz-student-cluster"
```

When prompted, press **Enter** to accept the default save path (`~/.ssh/id_ed25519`). You may optionally set a passphrase.

---

### 1.3 — Copy your key to the cluster

_Run on your local machine:_
```bash
$ ssh-copy-id <eth-username>@student-cluster.inf.ethz.ch
```
> Replace `<eth-username>` with your ETH username.

You will be asked for your ETH password **one last time** — after this step, no password is needed.

---

### 1.4 — (Optional) Add a connection shortcut

To connect with a short alias instead of the full hostname, run the following on your local machine:

_Run on your local machine:_
```bash
$ cat >> ~/.ssh/config << 'EOF'

Host eth-student-cluster
    HostName student-cluster.inf.ethz.ch
    User <eth-username>
    IdentityFile ~/.ssh/id_ed25519
EOF
```
> Replace `<eth-username>` on the `User` line with your ETH username.

You can then connect with:

_Run on your local machine:_
```bash
$ ssh eth-student-cluster
```

---

## 2 — Configure Conda on the Cluster

### 2.1 — Add conda init to `~/.bashrc`

Run the following command **once** on the cluster to automatically append the conda initialization block:

_Run on the cluster:_
```bash
$ cat >> ~/.bashrc << 'EOF'

# >>> conda initialize >>>
__conda_setup="$('/cluster/courses/cil/envs/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cluster/courses/cil/envs/etc/profile.d/conda.sh" ]; then
        . "/cluster/courses/cil/envs/etc/profile.d/conda.sh"
    else
        export PATH="/cluster/courses/cil/envs/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
EOF
```

---

### 2.2 — Reload your shell

_Run on the cluster:_
```bash
$ source ~/.bashrc
```

---

### 2.3 — (Optional) Set the default conda environment

_Run on the cluster:_
```bash
$ conda config --set default_activation_env /cluster/courses/cil/envs/envs/text-classification
```

---

### 2.4 — Activate the environment

_Run on the cluster:_
```bash
$ conda activate /cluster/courses/cil/envs/envs/text-classification
```

---

## 3 — Run Notebooks via VS Code

> This workflow runs Jupyter directly on the cluster — no GPU is allocated.  
> For GPU-enabled sessions, use [Jupyter Hub](https://student-jupyter.inf.ethz.ch/hub/spawn) instead.

**Step 1 — Start a Jupyter server on the cluster**

_Run on the cluster:_
```bash
$ jupyter lab --no-browser --port=8888
```

Copy the URL displayed in the terminal (e.g. `http://localhost:8888/?token=...`). You will need it in Step 5.

**Step 2 — Connect VS Code to the cluster**

In VS Code (local), connect via the **Remote – SSH** extension using `eth-student-cluster` (or your full hostname).

**Step 3 — Open your notebook**

Open the target `.ipynb` file in VS Code.

**Step 4 — Select a kernel**

Click **"Select Kernel"** in the top-right corner. Install the **Jupyter** and **Python** extensions if prompted.

**Step 5 — Connect to the Jupyter server**

Choose **"Existing Jupyter Server"** and paste the URL from Step 1.

**Step 6 — Choose the environment**

Select the **Python 3** kernel corresponding to the project's conda environment.