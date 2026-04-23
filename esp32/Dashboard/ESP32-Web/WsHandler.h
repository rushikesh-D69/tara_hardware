#ifndef WSHANDLER_H
#define WSHANDLER_H

#include <ESPAsyncWebServer.h>

void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client,
               AwsEventType type, void *arg, uint8_t *data, size_t len);

#endif // WSHANDLER_H
