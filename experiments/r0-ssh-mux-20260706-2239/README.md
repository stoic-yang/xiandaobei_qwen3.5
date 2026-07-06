# r0-ssh-mux-20260706-2239

- Intent: R0.0 harden SCNet SSH connection reuse before further probes.
- Change: `scripts/scnetctl.py` generated config now writes `ControlMaster auto`, `ControlPath ~/.ssh/cm-xiandaobei-%C`, and `ControlPersist 10m` into both `xiandaobei-login` and `xiandaobei-worker-auto` Host blocks.
- Validation: `python3 -m py_compile scripts/scnetctl.py` passed once.
- Regeneration: `python3 scripts/scnetctl.py attach` rewrote `/Users/keynary/.ssh/xiandaobei.generated.conf` for job `655597`, node `e03r2n07`, container IP `173.0.148.8`.

## Generated Config Evidence

Both Host blocks contain:

```sshconfig
ControlMaster auto
ControlPath ~/.ssh/cm-xiandaobei-%C
ControlPersist 10m
```

## Reuse Evidence

Because `xiandaobei-worker-auto` lives in the generated config, the verified command uses `-F /Users/keynary/.ssh/xiandaobei.generated.conf`.

```text
$ ssh -F /Users/keynary/.ssh/xiandaobei.generated.conf -O check xiandaobei-worker-auto
Master running (pid=77554)

$ /usr/bin/time -p ssh -F /Users/keynary/.ssh/xiandaobei.generated.conf xiandaobei-worker-auto true
real 15.69
user 0.02
sys 0.00

$ /usr/bin/time -p ssh -F /Users/keynary/.ssh/xiandaobei.generated.conf xiandaobei-worker-auto true
real 0.05
user 0.00
sys 0.00
```

The first command rebuilt the three-hop path; the second command reused the worker master and is below the 1s gate.

After writing this artifact, the worker master was explicitly prewarmed for the rest of the turn with `ssh -F /Users/keynary/.ssh/xiandaobei.generated.conf -MNf xiandaobei-worker-auto`; a follow-up `ssh ... true` measured `real 0.05`.

Note: bare `ssh -O check xiandaobei-worker-auto` does not load the generated config on this machine and returns `No ControlPath specified for "-O" command`; use `-F /Users/keynary/.ssh/xiandaobei.generated.conf` for this project.
