""" Allows endpoints to be imported as a module. """

from .asyncio.asyncio_runner import Endpoint as AsyncioEndpoint
from .asyncio.asyncio_runner import Job as AsyncioJob
# from .runner import Endpoint, Job
from .sxwl import Endpoint
