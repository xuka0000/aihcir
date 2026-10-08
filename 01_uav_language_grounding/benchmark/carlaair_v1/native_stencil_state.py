"""Preserve native CARLA semantic tags around differential AirSim stencils."""
import time


class NativeStencilState:
    BATCH_SIZE=32

    def __init__(self,client):
        self.client=client;self.values={};self.ephemeral=set();self.last_restore=None
        self.async_transport=callable(getattr(getattr(client,'client',None),'call_async',None))
        self.remember(client.simListSceneObjects(),False)
        if not self.values:raise RuntimeError('Native mesh stencil snapshot is empty')

    def calls(self,method,arguments):
        """Bound in-flight requests on one RPC event loop; drain every future."""
        arguments=list(arguments);results=[]
        if not self.async_transport:
            return [getattr(self.client,method)(*args) for args in arguments]
        for offset in range(0,len(arguments),self.BATCH_SIZE):
            pending=[];error=None
            try:
                for args in arguments[offset:offset+self.BATCH_SIZE]:
                    pending.append(self.client.client.call_async(method,*args))
            except Exception as exc:error=exc
            for future in pending:
                try:results.append(future.get())
                except Exception as exc:
                    if error is None:error=exc
            if error is not None:raise error
        return results

    def remember(self,names,ephemeral=True):
        matched=[];names=sorted(names)
        values=self.calls('simGetSegmentationObjectID',[(name.lower(),) for name in names])
        for name,value in zip(names,values):
            if not isinstance(value,int) or not 0<=value<=255:continue
            if name not in self.values:self.values[name]=value
            if ephemeral:self.ephemeral.add(name)
            matched.append(name)
        return matched

    def restore(self):
        started=time.monotonic();missing=[];present=None;changes=[];live=[]
        names=list(self.values)
        currents=self.calls('simGetSegmentationObjectID',[(name.lower(),) for name in names])
        for name,current in zip(names,currents):
            value=self.values[name]
            if current==-1:
                if present is None:present={n.lower() for n in self.client.simListSceneObjects()}
                if name.lower() not in present:
                    if name in self.ephemeral:missing.append(name);continue
                    raise RuntimeError('Original native mesh tag cannot be restored: '+name)
            live.append(name)
            if current!=value:changes.append(name)
        applied=self.calls('simSetSegmentationObjectID',[(name,self.values[name],False) for name in changes])
        for name,accepted in zip(changes,applied):
            if not accepted:
                raise RuntimeError('Original native mesh tag cannot be restored: '+name)
        readbacks=self.calls('simGetSegmentationObjectID',[(name.lower(),) for name in live])
        for name,value in zip(live,readbacks):
            if value!=self.values[name]:
                raise RuntimeError('Native mesh tag restoration readback differs: '+name)
        self.last_restore=dict(observed_at=time.time(),snapshot_mesh_owners=len(self.values),
            restored_mesh_owners=len(changes),previous_owned_actors_missing=len(missing),
            elapsed_seconds=time.monotonic()-started,same_thread_async_transport=self.async_transport,
            maximum_in_flight_requests=self.BATCH_SIZE if self.async_transport else 1,
            readback_mesh_owners=len(live),
            original_semantic_ID_readback_verified=True,ground_ray_labels_not_reclassified=True)
        return self.last_restore
