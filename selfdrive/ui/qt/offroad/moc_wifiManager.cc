/****************************************************************************
** Meta object code from reading C++ file 'wifiManager.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "wifiManager.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'wifiManager.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_WifiManager_t {
    QByteArrayData data[18];
    char stringdata0[211];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_WifiManager_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_WifiManager_t qt_meta_stringdata_WifiManager = {
    {
QT_MOC_LITERAL(0, 0, 11), // "WifiManager"
QT_MOC_LITERAL(1, 12, 13), // "wrongPassword"
QT_MOC_LITERAL(2, 26, 0), // ""
QT_MOC_LITERAL(3, 27, 4), // "ssid"
QT_MOC_LITERAL(4, 32, 13), // "refreshSignal"
QT_MOC_LITERAL(5, 46, 11), // "stateChange"
QT_MOC_LITERAL(6, 58, 9), // "new_state"
QT_MOC_LITERAL(7, 68, 14), // "previous_state"
QT_MOC_LITERAL(8, 83, 13), // "change_reason"
QT_MOC_LITERAL(9, 97, 14), // "propertyChange"
QT_MOC_LITERAL(10, 112, 9), // "interface"
QT_MOC_LITERAL(11, 122, 5), // "props"
QT_MOC_LITERAL(12, 128, 17), // "invalidated_props"
QT_MOC_LITERAL(13, 146, 11), // "deviceAdded"
QT_MOC_LITERAL(14, 158, 15), // "QDBusObjectPath"
QT_MOC_LITERAL(15, 174, 4), // "path"
QT_MOC_LITERAL(16, 179, 17), // "connectionRemoved"
QT_MOC_LITERAL(17, 197, 13) // "newConnection"

    },
    "WifiManager\0wrongPassword\0\0ssid\0"
    "refreshSignal\0stateChange\0new_state\0"
    "previous_state\0change_reason\0"
    "propertyChange\0interface\0props\0"
    "invalidated_props\0deviceAdded\0"
    "QDBusObjectPath\0path\0connectionRemoved\0"
    "newConnection"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_WifiManager[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   49,    2, 0x06 /* Public */,
       4,    0,   52,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    3,   53,    2, 0x08 /* Private */,
       9,    3,   60,    2, 0x08 /* Private */,
      13,    1,   67,    2, 0x08 /* Private */,
      16,    1,   70,    2, 0x08 /* Private */,
      17,    1,   73,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt,    6,    7,    8,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariantMap, QMetaType::QStringList,   10,   11,   12,
    QMetaType::Void, 0x80000000 | 14,   15,
    QMetaType::Void, 0x80000000 | 14,   15,
    QMetaType::Void, 0x80000000 | 14,   15,

       0        // eod
};

void WifiManager::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<WifiManager *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->wrongPassword((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: _t->refreshSignal(); break;
        case 2: _t->stateChange((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3]))); break;
        case 3: _t->propertyChange((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< const QVariantMap(*)>(_a[2])),(*reinterpret_cast< const QStringList(*)>(_a[3]))); break;
        case 4: _t->deviceAdded((*reinterpret_cast< const QDBusObjectPath(*)>(_a[1]))); break;
        case 5: _t->connectionRemoved((*reinterpret_cast< const QDBusObjectPath(*)>(_a[1]))); break;
        case 6: _t->newConnection((*reinterpret_cast< const QDBusObjectPath(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (WifiManager::*)(const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&WifiManager::wrongPassword)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (WifiManager::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&WifiManager::refreshSignal)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject WifiManager::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_WifiManager.data,
    qt_meta_data_WifiManager,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *WifiManager::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *WifiManager::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_WifiManager.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int WifiManager::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void WifiManager::wrongPassword(const QString & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void WifiManager::refreshSignal()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
