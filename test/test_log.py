from torch4keras.snippets import LoggerHandler
import time
log = LoggerHandler(log_path='./log/log.log', handles=['StreamHandler', 'TimedRotatingFileHandler'],
                    handle_config={'when': 'S', 'interval': 3, 'backupCount': 2, 'encoding': 'utf-8'})

log.debug('debug')
log.info('info')
log.warning('warning')
log.error('error')
log.critical('critical')

for i in range(10):
    log.info('hello '+str(i))
    time.sleep(2)