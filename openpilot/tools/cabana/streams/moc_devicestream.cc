/****************************************************************************
** Meta object code from reading C++ file 'devicestream.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "devicestream.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'devicestream.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_DeviceStream_t {
    QByteArrayData data[1];
    char stringdata0[13];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DeviceStream_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DeviceStream_t qt_meta_stringdata_DeviceStream = {
    {
QT_MOC_LITERAL(0, 0, 12) // "DeviceStream"

    },
    "DeviceStream"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DeviceStream[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void DeviceStream::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

QT_INIT_METAOBJECT const QMetaObject DeviceStream::staticMetaObject = { {
    QMetaObject::SuperData::link<LiveStream::staticMetaObject>(),
    qt_meta_stringdata_DeviceStream.data,
    qt_meta_data_DeviceStream,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *DeviceStream::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DeviceStream::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DeviceStream.stringdata0))
        return static_cast<void*>(this);
    return LiveStream::qt_metacast(_clname);
}

int DeviceStream::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = LiveStream::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_OpenDeviceWidget_t {
    QByteArrayData data[1];
    char stringdata0[17];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_OpenDeviceWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_OpenDeviceWidget_t qt_meta_stringdata_OpenDeviceWidget = {
    {
QT_MOC_LITERAL(0, 0, 16) // "OpenDeviceWidget"

    },
    "OpenDeviceWidget"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_OpenDeviceWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void OpenDeviceWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

QT_INIT_METAOBJECT const QMetaObject OpenDeviceWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<AbstractOpenStreamWidget::staticMetaObject>(),
    qt_meta_stringdata_OpenDeviceWidget.data,
    qt_meta_data_OpenDeviceWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *OpenDeviceWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *OpenDeviceWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_OpenDeviceWidget.stringdata0))
        return static_cast<void*>(this);
    return AbstractOpenStreamWidget::qt_metacast(_clname);
}

int OpenDeviceWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractOpenStreamWidget::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
