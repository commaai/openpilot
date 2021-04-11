import QtQuick 2.0

Item {
  id: root
  signal qmlSignal()

  Flickable {
    id: flickArea
    objectName: "flickArea"
    anchors.fill: parent
    contentHeight: helpText.height
    contentWidth: helpText.width
    flickableDirection: Flickable.VerticalFlick
    flickDeceleration: 7500.0
    maximumFlickVelocity: 10000.0
    pixelAligned: true

    onAtYEndChanged: root.qmlSignal()

    Text {
      id: helpText
      width: flickArea.width
      font.pixelSize: font_size
      textFormat: Text.RichText
      color: "white"
      wrapMode: Text.Wrap

      text: text_view
    }
  }
}


