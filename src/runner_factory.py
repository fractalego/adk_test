from typing import Optional

from google.adk import Runner
from google.adk.sessions import InMemorySessionService


class RunnerFactory:
    def __init__(self, app_name: str, user_id: str, session_id: str, initial_state: Optional[dict] = None):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id
        self._initial_state = initial_state
        self._session_service = None

    async def get_runner(self, agent) -> 'Runner':
        session_service = await self.get_session_service()
        return Runner(
            agent=agent,
            app_name=self.app_name,
            session_service=session_service
        )

    async def get_session_service(self) -> 'Session':
        if not self._session_service:
            self._session_service = InMemorySessionService()
            await self._session_service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id,
                state=self._initial_state
            )
        return self._session_service
