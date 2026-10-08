"""One bounded SSH attempt to an authenticated, locally owned endpoint."""
def connect_once(remote):
    import paramiko
    from run_public_mcq_teacher_autodl import system_proxy_socket
    remote.heartbeat();remote.close()
    snapshot=remote.api.connection()
    client=paramiko.SSHClient()
    client.load_host_keys(str(remote.api.token_path.parent/'known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    stream=None
    try:
        stream=system_proxy_socket(snapshot['proxy_host'],int(snapshot['ssh_port']))
        client.connect(snapshot['proxy_host'],port=int(snapshot['ssh_port']),username='root',password=snapshot['root_password'],
                       sock=stream,timeout=15,banner_timeout=15,auth_timeout=15,look_for_keys=False,allow_agent=False)
        remote.client=client;remote.proxy_socket=stream
    except BaseException:
        client.close()
        if stream is not None:stream.close()
        raise
    finally:del snapshot
